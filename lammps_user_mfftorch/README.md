# LAMMPS on Ascend

The plain `mff/torch` pair style runs a persistent eager MACE-ICTC model per MPI
rank. Ascend C evaluates model operators and packs force and feature buffers;
HCCL transfers those buffers between NPUs. LAMMPS retains CPU integration,
neighbor lists, domain decomposition and atom migration.

Read the [build and usage guide](docs/ASCEND.md) for installation, FP32 checkpoint
requirements, MPI/HCCL launch and supported observables. Converted MACE-MH-1
archives use the same interface; see [checkpoint conversion](../docs/CONVERSION.md).

The library includes the Ascend pair style, installer and build configuration.
