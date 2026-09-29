# Maintained experiment environment

Use CPython 3.13 on Windows x64 and the root `requirements.txt`. This single
file includes all dependencies for synthetic and MOPTA08 runs, optional
algorithms such as CMA-ES, and tests.
JAX/NumPyro are included in the main requirements because the shared MCTS-VS
module imports its SAAS component for all of its optimizer variants.

Upstream reference sources retain their original licences and provenance in
`UPSTREAM_SOURCES.md` and `upstream_sources.json`. Historical upstream
requirements are not installation instructions for the maintained native
adapters. No NAS, MuJoCo, LassoBench or Rover dependency is required by the
released run plans. MOPTA08 requires the separately acquired, checksum-verified
Windows evaluator described in the root README.
