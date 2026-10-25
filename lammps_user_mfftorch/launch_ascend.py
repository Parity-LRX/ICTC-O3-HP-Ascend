"""Set Ascend dispatch/NUMA options before starting LAMMPS or its MPI launcher.

Run this standard-library-only script inside the intended compute allocation.
Explicit CPU_AFFINITY_CONF and TASK_QUEUE_ENABLE settings take precedence.
"""
import argparse
import json
import os
import re
import subprocess
import sys


def indices(text):
    result = []
    for item in text.split(','):
        match = re.fullmatch(r'(\d+)(?:-(\d+))?', item.strip())
        if match is None:
            raise ValueError(f'Unsupported device or CPU range: {item}')
        first = int(match[1])
        last = first if match[2] is None else int(match[2])
        if last < first:
            raise ValueError(f'Reversed device or CPU range: {item}')
        result.extend(range(first, last + 1))
    if len(set(result)) != len(result):
        raise ValueError('Device and CPU lists must not repeat indices')
    return result


def topology_affinity(text, visible, allowed):
    """Divide each actual near-CPU group between its visible devices."""
    topology = {}
    for line in text.splitlines():
        match = re.match(r'^NPU(\d+)\s+', line.lstrip())
        if match and re.fullmatch(r'\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*', line.split()[-1]):
            topology[int(match[1])] = tuple(sorted(set(indices(line.split()[-1])) & set(allowed)))
    devices = sorted(topology) if visible is None else indices(visible)
    if not devices:
        raise ValueError('No NPU topology entries were found')
    groups = {}
    for logical, physical in enumerate(devices):
        cores = topology.get(physical)
        if not cores:
            raise ValueError(f'NPU {physical} has no near CPUs in this allocation')
        groups.setdefault(cores, []).append(logical)
    chosen, occupied = {}, set()
    for cores, members in groups.items():
        if len(cores) < len(members):
            raise ValueError('The allocation has too few near CPUs for distinct NPU bindings')
        for ordinal, device in enumerate(members):
            start = len(cores) * ordinal // len(members)
            stop = len(cores) * (ordinal + 1) // len(members)
            assigned = cores[start:stop]
            if occupied.intersection(assigned):
                raise ValueError('Overlapping near-CPU groups need an explicit CPU_AFFINITY_CONF')
            runs = []
            for core in assigned:
                if runs and core == runs[-1][-1] + 1:runs[-1].append(core)
                else:runs.append([core])
            # TorchNPU custom bindings accept a single closed interval per NPU.
            selected = max(runs, key=len)
            chosen[device] = selected[0], selected[-1]
            occupied.update(assigned)
    return '1,' + ','.join(f'npu{device}:{first}-{last}'
                          for device, (first, last) in sorted(chosen.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execution', choices=('eager', 'graph'),
                        help='Select persistent eager execution or ACLGraph replay')
    parser.add_argument('--queue', choices=('0', '1', '2'),
                        help='Override TASK_QUEUE_ENABLE; default is 2 for eager execution')
    parser.add_argument('--affinity', choices=('auto', 'none'), default='auto',
                        help='Bind using actual NPU/CPU topology unless already configured')
    parser.add_argument('--dry-run', action='store_true', help='Print the command and selected environment')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command[:1] == ['--']:command = command[1:]
    if not command:parser.error('Supply a LAMMPS or MPI command after --')
    environment = dict(os.environ)
    if args.execution is not None:environment['MFF_ASCEND_EXECUTION'] = args.execution
    if args.queue is not None:environment['TASK_QUEUE_ENABLE'] = args.queue
    else:environment.setdefault('TASK_QUEUE_ENABLE',
                                '1' if environment.get('MFF_ASCEND_EXECUTION') == 'graph' else '2')
    if args.affinity == 'auto' and not environment.get('CPU_AFFINITY_CONF'):
        if not hasattr(os, 'sched_getaffinity'):
            parser.error('Automatic affinity requires Linux; use --affinity none')
        try:
            result = subprocess.run(['npu-smi', 'info', '-t', 'topo'], check=True,
                                    capture_output=True, text=True, timeout=20)
            environment['CPU_AFFINITY_CONF'] = topology_affinity(result.stdout,
                environment.get('ASCEND_RT_VISIBLE_DEVICES'), os.sched_getaffinity(0))
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            parser.error(f'Automatic affinity failed: {error}. Set CPU_AFFINITY_CONF or use --affinity none.')
    selected = {key:environment.get(key) for key in (
        'MFF_ASCEND_EXECUTION', 'TASK_QUEUE_ENABLE', 'CPU_AFFINITY_CONF', 'ASCEND_RT_VISIBLE_DEVICES')}
    if args.dry_run:
        print(json.dumps(dict(command=command, environment=selected), indent=2))
        return
    print('Ascend launch: ' + json.dumps(selected), file=sys.stderr, flush=True)
    os.execvpe(command[0], command, environment)


if __name__ == '__main__':main()
