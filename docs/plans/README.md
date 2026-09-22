# Design plans (historical)

Planning documents written **before** the work they describe was implemented.
They are kept for the rationale and for the "why not the obvious approach"
notes, which the final code does not record.

**All five plans below are implemented and verified end-to-end** — every
construct they propose is covered by a passing demo in the e2e suite (see the
*Implemented language surface* tables in the top-level
[README](../../README.md)). Do not read them as a to-do list; read them as a
record of the design decisions and the alternatives that were rejected.

| Plan | Subject | Verified by |
|------|---------|-------------|
| [`PLAN_builtin_registry.md`](PLAN_builtin_registry.md) | One shared `BuiltinRegistry` replacing three duplicated builtin name lists | `lib/Frontend/BuiltinRegistry.*` |
| [`PLAN_mangle.md`](PLAN_mangle.md) | One `deviceMangledName`/`mangleScopeName` replacing five duplicate mangle sites | `lib/Frontend/Mangle.h` |
| [`PLAN_overload.md`](PLAN_overload.md) | `__device__` overload resolution (`conversionRank`/`resolveOverload`) | `test/overload.vc`, `test/const_overload.vc` |
| [`PLAN_f16_ssbo.md`](PLAN_f16_ssbo.md) | f16 storage-buffer scalar load/store on the MLIR path | `test/f16_ssbo.vc` |
| [`PLAN_wmma.md`](PLAN_wmma.md) | `wmma::*` → `spirv.KHR.CooperativeMatrix` (cooperative matrix) | `test/wmma_gemm.vc` (MLIR only) |
