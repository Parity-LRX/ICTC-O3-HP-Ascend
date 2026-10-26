"""Compose standalone O(3) layers and differentiate on an allocated Ascend NPU."""
import argparse
import math

import torch
import torch_npu

from ictc_o3 import (
    PackedGraphMessage,
    channel_linear,
    create_symmetric_contraction,
    direction_harmonics,
    normalized_silu,
    species_channel_linear,
)
from ictc_o3.ascendc import enable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, default=128)
    parser.add_argument("--edges", type=int, default=1024)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--lmax", type=int, default=2)
    parser.add_argument("--elements", type=int, default=3)
    parser.add_argument("--correlation", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--edge-chunk-size", type=int, default=16384)
    parser.add_argument("--ascendc-mode", choices=("performance", "capacity", "native"), default="performance")
    args = parser.parse_args()
    if min(args.nodes, args.edges, args.channels, args.elements, args.edge_chunk_size) < 1 or args.lmax < 0:
        parser.error("Sizes must be positive and lmax must be nonnegative")

    torch.npu.set_device(0)
    enable(mode=args.ascendc_mode)
    device, channels = "npu", args.channels
    message = PackedGraphMessage.create(
        channels=channels, input_lmax=args.lmax, edge_lmax=args.lmax,
        output_lmax=args.lmax, edge_chunk_size=args.edge_chunk_size,
        device=device,
    )
    features = torch.randn(args.nodes, channels * (args.lmax + 1)**2,
                           device=device, requires_grad=True)
    directions = torch.randn(args.edges, 3, device=device, requires_grad=True)
    unit = torch.nn.functional.normalize(directions, dim=-1)
    harmonics = torch.cat(direction_harmonics(unit, args.lmax, policy="generated"), -1)
    gates = torch.randn(args.edges, message.gate_width, device=device, requires_grad=True)
    src = torch.randint(args.nodes, (args.edges,), device=device)
    dst = torch.randint(args.nodes, (args.edges,), device=device)
    species = torch.randint(args.elements, (args.nodes,), device=device)
    attributes = torch.nn.functional.one_hot(species, args.elements).float()

    # Merge path multiplicities with a shared channel matrix for each degree.
    blocks = message.unpack_output(message(features, harmonics, gates, src, dst))
    linears = torch.nn.ParameterList([
        torch.nn.Parameter(torch.randn(channels, blocks[l].shape[1], device=device)
                           / math.sqrt(blocks[l].shape[1]))
        for l in range(args.lmax + 1)
    ])
    # Symmetric contraction uses degree-major [channel, angular] blocks.
    hidden = torch.cat([
        channel_linear(blocks[l], linears[l]).flatten(1)
        for l in range(args.lmax + 1)
    ], -1)
    # Scale this synthetic graph input before its higher-order polynomial.
    hidden = hidden / max(1., args.edges / args.nodes)
    contraction = create_symmetric_contraction(
        num_elements=args.elements, channels=channels, lmax=args.lmax,
        target_lmax=0, correlation=args.correlation, device=device, training=True,
    ).train()
    scalar = contraction(hidden, attributes, node_type_idx=species)
    bank = torch.nn.Parameter(torch.randn(args.elements, 1, channels, device=device)
                              / math.sqrt(channels))
    readout = species_channel_linear(scalar.unsqueeze(-1), bank, species).squeeze(-1)
    energy = normalized_silu(readout).sum()
    parameters = [*linears, *contraction.parameters(), bank]
    optimizer = torch.optim.AdamW(parameters, lr=1.e-3)

    # A force-like gradient followed by parameter backward exercises mixed
    # second derivatives; directions here are synthetic, not atomic positions.
    direction_gradient, = torch.autograd.grad(energy, directions, create_graph=True)
    loss = energy.square() / args.nodes**2 + direction_gradient.square().mean()
    loss.backward()
    optimizer.step()
    torch.npu.synchronize()
    print("Message:", (args.nodes, message.output_width), "Scalars:", tuple(scalar.shape))
    print("Loss:", float(loss.detach().cpu()), "Direction gradient:", tuple(direction_gradient.shape))


if __name__ == "__main__":
    main()
