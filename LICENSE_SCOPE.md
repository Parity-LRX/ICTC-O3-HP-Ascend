# License scope — ICTC-O3-HP-Ascend 0.1.0

This independent Ascend port retains the upstream license scopes below.
The new `mace_ictc/ascend/` backend, `ictc_o3/` API and their integration changes
follow the project-owned HP implementation scope. Upstream provenance is
recorded in [UPSTREAM.md](UPSTREAM.md).

This release offers the newly added, project-owned HP implementations under
the unmodified PolyForm Noncommercial License 1.0.0 in [LICENSE](LICENSE).
Rights holders can separately grant commercial permissions according to
[COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md). That document is a procedure,
not an automatic commercial license. The standard PolyForm exceptions remain.

## New HP implementations

The PolyForm offer covers the following new HP components and their newly
added integration changes, to the extent their copyright is controlled by
the project rights holders:

- `mace_ictc/hp/`: model construction and whole-model APIs.
- `ictc_o3/` and `mace_ictc/ascend/`: Ascend operator APIs, schedules and adjoints.
- `mace_ictc/ascendc/`: native Ascend C kernels, UB tiling and Python bindings.
- `generated_direction_harmonics.py`, `factorized_contraction.py`,
  `sparse_weighted_product.py`, `reference_u_residency.py`,
  `adjoint_requirements.py`, `operator_plans.py` and `fp32_policy.py` in
  `mace_ictc/models/`: fixed mathematical plans, direction adjoints and
  reference/runtime integration.
- Newly added foundation checkpoint adapters and Ascend integration portions.
- Newly added HP selection/layout integration portions of the existing model,
  training, checkpoint and export files, plus newly created HP usage examples.

The module/path layout itself does not remove pre-existing third-party rights.
Where a listed component includes pre-existing permissive material, its original
copyright and permission notices continue to apply to that material.

## Preserved MIT rights

The earlier MACE-ICTC implementation remains covered by its existing MIT offer,
reproduced in [LICENSES/MACE-ICTC-MIT.txt](LICENSES/MACE-ICTC-MIT.txt). Existing
MIT portions in modified files retain those permissions. This release does
not retroactively revoke earlier recipients' MIT rights or claim that merely
renaming the package makes previously MIT-licensed code noncommercial.

In particular, the original `ictd_irreps.py`, `pure_cartesian_ictd_fix.py`,
`csrc/ictd_tp.cpp` and `csrc/ictd_tp_cuda.cu` baseline material keeps its MIT
permissions; new HP additions are distinguished by the scope above.

`models/_mace_cg.py` and `models/_mace_symmetric_contraction.py` retain their
MACE author notices and MIT permission. [LICENSES/MACE-MIT.txt](LICENSES/MACE-MIT.txt)
contains the upstream MACE notice. The e3nn notice is in
[LICENSES/e3nn-MIT.txt](LICENSES/e3nn-MIT.txt); use of e3nn as a dependency does
not replace that dependency's license with PolyForm. All third-party source
notices, patents and model-weight licenses retain their original scope.

Package metadata uses `PolyForm-Noncommercial-1.0.0 AND MIT` to describe the
standard noncommercial distribution route for this combined Python edition.
It does not mean that every source line is relicensed under both terms, or
that MIT alone licenses the newly restricted HP components. An alternative
commercial grant can cover only rights that its issuer is entitled to grant.

## LAMMPS integration

The separate `lammps_user_mfftorch` integration is not automatically relicensed
as PolyForm by this Python release. Existing rights and third-party notices
are retained. LAMMPS itself is GPLv2; interface code, linking and distribution
of a combined executable or required runtime libraries need their own license
compatibility assessment. This package does not assert that a PolyForm grant
or a commercial Python grant automatically permits any GPL combination.
See the [official LAMMPS license guidance](https://docs.lammps.org/Intro_opensource.html).

This release provides source and license texts, not a new license for third-party
training data or pretrained potentials. Rights holders should confirm institutional
and contributor authority before publishing or signing commercial agreements.
