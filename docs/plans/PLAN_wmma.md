# P3B — WMMA / Tensor-Core (MLIR backend, Vulkan-native cooperative matrix)

## Goal

Minimal viable CUDA `nvcuda::wmma`-style surface for a tensor-core GEMM demo,
lowered through the MLIR/SPIR-V backend to `spirv.KHR.CooperativeMatrix*`.
GLSL backend deferred (glslc instability). One warp computes one MxN output
tile: D = A*B + C, host verifies D == A*B.

**SHAPE DECISION (per user, 2026-09-17):** expose Vulkan-native shapes, NOT
CUDA's hardcoded 16x16x16. Probed this dev GPU: supports VK_KHR_cooperative_matrix
(compute stage, cooperativeMatrix=true) but only `8x8x16` shapes — incl.
`8x8x16 fp16-input → fp32-accumulate` (A/B=FLOAT16, C/D=FLOAT32, scope=Subgroup).
**16x16x16 is NOT supported by this GPU.** So the fragment type carries real
M/N/K dims (from the device query), and the P3B demo uses 8x8x16. The API
keeps the CUDA WMMA `wmma::fragment<use, M, N, K, T, layout>` spelling so source
stays portable; Sema validates the (M,N,K,T) combo against a device-shape table
baked from the runtime query (or a compile-time default of 8x8x16 for this GPU).

Verified source facts (from research agents that read mma.h + LLVM18 SPIR-V headers):

### CUDA WMMA (crt/mma.h, reference for API SHAPE — dims differ on Vulkan)
- `fragment<Use, M, N, K, T, Layout=void>` — Use ∈ {matrix_a, matrix_b, accumulator}.
  A/B take a Layout template param (`row_major` / `col_major` TYPES); accumulator
  has NO Layout (5 template args).
- CUDA's base fp16 shape is (16,16,16) — **NOT supported on this GPU**. We keep
  the API spelling but the dims come from the Vulkan device query. P3B default
  demo shape: (8,8,16) fp16→fp32.
- `load_matrix_sync(frag&, const T* p, unsigned ldm)` — 3 args for A/B (layout in type).
  `load_matrix_sync(accum&, const T* p, unsigned ldm, layout_t layout)` — 4 args for
  accumulator (runtime `mem_row_major`/`mem_col_major` enum).
- `store_matrix_sync(T* p, const accum_frag&, unsigned ldm, layout_t layout)` — accumulator
  ONLY, non-const ptr, runtime layout_t, no default.
- `mma_sync(accum& d, const a_frag& a, const b_frag& b, const accum& c)` — d = a*b+c,
  d may alias c (in-place accumulate is idiomatic). 4 args, no satf on fp16/fp32.
- Storage counts are shape-dependent and warp-distributed/opaque; we do NOT expose
  `.x[]` element access (lower load/store/mma directly to coopmatrix ops).

### MLIR SPIR-V CooperativeMatrix (LLVM 18 headers)
- Type: `!spirv.coopmatrix<rows x cols x elem, scope, use>` — NO `KHR` prefix on type
  (the `!spirv.KHR.coopmatrix` in .td doc comments is a TYPO).
- use keywords: `MatrixA`, `MatrixB`, `MatrixAcc` (NOT MatrixAccumulator).
- scope: `Subgroup`.
- 8x8x16 (this GPU's shape): A=`!spirv.coopmatrix<8x8xf16, Subgroup, MatrixA>`,
            B=`!spirv.coopmatrix<16x8xf16, Subgroup, MatrixB>` (rows=K=16, cols=N=8),
            C/D=`!spirv.coopmatrix<8x8xf32, Subgroup, MatrixAcc>`.
  NOTE: per SPIR-V MulAdd semantics, A is M×K, B is K×N, C/D is M×N. For (M,N,K)=(8,8,16):
  A=8x16, B=16x8, C/D=8x8. The `rows x cols` in the type are A's rows×K, B's K×N, etc.
- `spirv.KHR.CooperativeMatrixLoad %ptr, %stride, <RowMajor> : ptr_ty, i32 -> cm_ty`
  — stride is an SSA integer OPERAND; layout (`RowMajor`/`ColumnMajor`) is an
  ATTRIBUTE in `<...>`; memory_operand optional attr.
- `spirv.KHR.CooperativeMatrixStore %ptr, %obj, %stride, <RowMajor> : ptr_ty, cm_ty, i32`
  — operand order: pointer, object, stride, layout.
- `%d = spirv.KHR.CooperativeMatrixMulAdd %a, %b, %c : tyA, tyB -> tyC`
  — d = a*b+c; result type == %c type (AllTypesMatch); a/b may differ from c (f16 vs f32).
- Load/Store pointer pointee must be SCALAR (or vector): `!spirv.ptr<f16, StorageBuffer>`.
  Index into SSBO array with spirv.AccessChain first, pass the scalar-pointee ptr to Load.
- target_env: SPIR-V **1.6** + `CooperativeMatrixKHR` capability (6022) +
  `SPV_KHR_cooperative_matrix` extension. Vulkan 1.2 core.
- No turnkey vector.contract/gpu.mma→coopmatrix lowering in this build → emit
  spirv.KHR.CooperativeMatrix* DIRECTLY from ASTToMLIR (same as other spirv ops).

### Compiler extension points (already located)
- AST.h:63 `enum class TypeKind { Builtin, Pointer, Reference, Vector, Record, Typedef }`
  → add `WmmaFragment`.
- AST.h:145 ASTNode::NodeKind — no new node kinds needed; reuse CallExpr (resolvedCallee
  path from P3A) for load/store/mma/fill, since they're function calls syntactically.
- Lexer.cpp:122 already has `kw_wmma`. Need layout/use/precision keywords too.
- Parser.cpp:1020 parseType, 1760 parsePrimary — fragment type parse + call parse.
- Sema.cpp — type-check fragment decls + wmma call arg types.
- VCToGPU.cpp:54 getVCTargetEnv, :365 `vc.uses_subgroup` read → add parallel
  `vc.uses_coopmatrix` channel bumping to V_1_6 + CooperativeMatrixKHR + ext.
- ASTToMLIR.cpp:4766 sets `vc.uses_subgroup` → set `vc.uses_coopmatrix` too.
- VCRuntime.cpp:377-423 device feature pNext chain → add
  VkPhysicalDeviceCooperativeMatrixFeaturesKHR + VK_KHR_COOPERATIVE_MATRIX extension.

## Design decisions (locked)

1. **Fragment type**: new `TypeKind::WmmaFragment` with fields:
   - `enum class WmmaUse { MatrixA, MatrixB, Accumulator }`
   - `enum class WmmaPrecision { F16, F32 }`
   - `enum class WmmaLayout { RowMajor, ColMajor, None }` (None for accumulator)
   - `unsigned M, N, K` — real tile dims from source `fragment<use,M,N,K,T,layout>`.
     P3B demo uses M=8,N=8,K=16. No hardcoded 16x16x16.
2. **Parse fragment dims**: `wmma::fragment < use , M , N , K , precision , layout? >`
   parses the literal M/N/K integers and stores them on the type. Sema validates
   (M,N,K,precision) is a shape the target supports (P3B: 8x8x16 fp16→fp32; reject
   16x16x16 with a clear "shape not supported by target" error). The shape table is
   a compile-time constant for now (8x8x16 fp16→f32); a future P3C can feed it from
   the runtime device query via a `-DVC_COOP_MATRIX_SHAPE` define or target_env attr.
3. **Scope handling**: `wmma::fragment<...>` is a qualified type. Parse `wmma` keyword,
   expect `::`, then `fragment< use, dims, dims, dims, precision, layout? >`.
   `wmma::load_matrix_sync(...)` etc. parse as scoped CallExpr (existing `ns::f` path).
4. **fragment variable storage**: a `wmma::fragment` local decl emits NO storage in
   SPIR-V — the coopmatrix value is SSA, materialized at load, consumed at mma/store.
   So `wmma::fragment<...> a;` is a forward-declared slot that only gets a value when
   load_matrix_sync / fill_fragment writes it. Model as a scoped SSA value (like a
   normal local) whose MLIR type is the coopmatrix type. Assignments via the wmma
   intrinsics write the slot; reads (mma operands, store operand) read it.
5. **fill_fragment(c_frag, 0.0f)** → `spirv.KHR.CooperativeMatrixLoad` from a constant?
   NO — there's no "fill" coopmatrix op. Lower to a MulAdd with a zero matrix, OR
   simpler: skip fill, require explicit load before mma (idiomatic CUDA fills then
   overwrites via load anyway in the tiled loop). For the single-tile demo, the
   accumulator is loaded from C (D=A*B+C), so fill isn't needed. Support fill_fragment
   by lowering to a load of a zero-initialized shared/global buffer OR document as
   "load C first". Decision: **implement fill_fragment via a runtime-zeroed scratch
   SSBO load** is overkill for P3B; instead lower fill_fragment to a no-op-style
   spirv.Constant-based composite? coopmatrix has no CompositeConstruct from scalars.
   **SIMPLEST**: require the demo to `load_matrix_sync(C, ...)` the accumulator (D=A*B+C
   semantics), and implement fill_fragment as an error/TODO if used on the path that
   doesn't load. Revisit: actually for a pure GEMM (C=0) the demo can load a zeroed C
   buffer. So fill_fragment NOT in P3B minimal set — drop it from the API subset.
   API subset = {fragment, load_matrix_sync, store_matrix_sync, mma_sync}. (User chose
   "minimal viable set" — fill was listed but is the lowest priority; drop to keep
   scope tight. Document as P3C.)
6. **target_env bump is additive**: existing kernels stay V_1_0/1.3; only modules
   that set `vc.uses_coopmatrix` get V_1_6. The two flags are independent (a module
   could use both subgroup + coopmatrix → V_1_6 + all caps).
7. **GLSL backend**: skip lowering; if a kernel uses wmma and is compiled via vcc
  (GLSL), emit a clear error "wmma requires the MLIR backend (-emit=full)". Don't
  crash silently.

## Implementation steps

### Step 1 — AST: WmmaFragment type (include/vc/Frontend/AST.h)
- Add `WmmaFragment` to `TypeKind`.
- Add enums `WmmaUse`, `WmmaPrecision`, `WmmaLayout` (in vc namespace, near BuiltinTypeKind).
- `class WmmaFragmentType : public Type` with `WmmaUse use; WmmaPrecision prec;
  WmmaLayout layout;` ctor + classof. (M=N=K=16 implicit.)
- No new ASTNode kinds.

### Step 2 — Lexer: WMMA keywords (lib/Frontend/Lexer.cpp + Lexer.h)
- `kw_wmma` exists. Add: `kw_fragment`, `kw_matrix_a`, `kw_matrix_b`, `kw_accumulator`,
  `kw_row_major`, `kw_col_major`. (precision reuses kw_half/kw_float; layout_t enum
  values `mem_row_major`/`mem_col_major` for accumulator load/store can be parsed as
  identifiers and matched by Sema — no new keywords needed for those, OR add
  kw_mem_row_major/kw_mem_col_major. Decision: parse as identifiers, Sema matches
  string — keeps keyword bloat down.)
- Register in the keyword table.

### Step 3 — Parser: fragment type + scoped wmma calls (lib/Frontend/Parser.cpp)
- `parseType`: when current token is `kw_wmma` followed by `::` `fragment`, parse
  `wmma::fragment < use , 16 , 16 , 16 , precision , layout? >` → build
  WmmaFragmentType. The `<>` angle-bracket arg list is new syntax for VC types
  (vectors use float4 suffix, not templates) — add a small dedicated parser, don't
  try to generalize C++ template parsing.
- Scoped calls `wmma::load_matrix_sync(...)`: the existing `ns::f()` parse path should
  already produce a scoped callee name; verify it handles `wmma::load_matrix_sync`.
  If not, extend the qualified-name parser to recognize `wmma` as a known namespace.
- `wmma::fragment<...> a;` variable decl: parseType returns WmmaFragmentType, then
  parseVarDecl handles it like any typed local. The decl just needs a name + type;
  no initializer (fragments are initialized by load/mma).

### Step 4 — Sema: type-check wmma (lib/Frontend/Sema.cpp, lib/Frontend/Sema.h)
- Recognize wmma call sites by callee name:
  - `wmma::load_matrix_sync(frag, ptr, ldm[, layout])` — arg0 must be a fragment
    local (A/B: 3 args; accumulator: 4 args with layout). Check ptr is `T*` matching
    fragment precision, ldm is int.
  - `wmma::store_matrix_sync(ptr, accum_frag, ldm, layout)` — 4 args, arg1 must be
    accumulator fragment.
  - `wmma::mma_sync(d, a, b, c)` — 4 args, d & c accumulator-fp32, a matrix_a-fp16,
    b matrix_b-fp16. d and c may be the same variable (in-place).
- Mark these calls so the backend knows to lower them (a flag on CallExpr, or the
  backend matches the `wmma::` callee string). Decision: backend matches callee
  string — no AST change. Keep Sema checks, don't set resolvedCallee (these aren't
  FunctionDecls).
- Set `module` flag `usesCoopmatrix` (mirror usesSubgroup) — actually this is an
  ASTToMLIR concern; Sema just validates. ASTToMLIR sets the module attr when it
  emits a coopmatrix op.

### Step 5 — MLIR backend: emit spirv.KHR.CooperativeMatrix* (lib/Codegen/ASTToMLIR.cpp)
- Fragment local decl → allocate an SSA slot of coopmatrix type (or just track the
  most recent value bound to the name, since fragments aren't aliased/reused oddly
  in the minimal demo — but to be safe use a slot: a region value that load/fill
  writes and mma/store reads, lowered via spirv.Variable + Load/Store of coopmatrix
  values). Decision: **use a simple name→Value map with phi-style reassignment**
  like other locals; coopmatrix values are first-class SSA values, store the latest
  in the local symbol table. load_matrix_sync produces a new coopmatrix Value bound
  to the fragment name; mma_sync consumes a/b/c and binds d; store consumes the
  fragment. This matches how the backend already treats scalar locals.
- `wmma::load_matrix_sync(frag, ptr, ldm, [layout])`:
  - Compute the scalar-pointee pointer: the SSBO arg is already a `!spirv.ptr<...,StorageBuffer>`;
    AccessChain to element 0 (or the tile offset) → `!spirv.ptr<f16, StorageBuffer>`.
    For the single-tile demo, tile offset is 0 (one warp, one tile at the buffer start).
  - layout: A/B from the fragment's WmmaLayout field; accumulator from the runtime
    `mem_row_major`/`mem_col_major` arg → map to `<RowMajor>`/`<ColumnMajor>`.
  - stride: ldm arg as i32 constant/SSA value.
  - Emit `spirv.KHR.CooperativeMatrixLoad` → bind result to frag name.
- `wmma::mma_sync(d, a, b, c)`: emit `spirv.KHR.CooperativeMatrixMulAdd %a, %b, %c`
  → bind result to d name (may equal c name → rebind, fine).
- `wmma::store_matrix_sync(ptr, frag, ldm, layout)`: emit
  `spirv.KHR.CooperativeMatrixStore %ptr, %frag, %ldm, <layout>`.
- On first coopmatrix op emitted, set module attr `vc.uses_coopmatrix`.
- coopmatrix type builder: `spirv::CooperativeMatrixType::get(elemTy, rows, cols,
  spirv::Scope::Subgroup, spirv::CooperativeMatrixUseKHR::MatrixA/B/Acc)`.
  rows/cols per use for (M,N,K)=(8,8,16): A=8x16, B=16x8, C/D=8x8. elemTy: f16 for
  A/B, f32 for C/D.

### Step 6 — target_env + capability (lib/Codegen/VCToGPU.cpp)
- `getVCTargetEnv(ctx, usesSubgroup, usesCoopmatrix)`: when usesCoopmatrix,
  version = V_1_6, add `CooperativeMatrixKHR` capability, add
  `SPV_KHR_cooperative_matrix` extension. (If both subgroup+coopmatrix, V_1_6
  supersedes V_1_3, keep all caps.)
- Read `module->hasAttr("vc.uses_coopmatrix")` at :365, pass through.

### Step 7 — Runtime: enable cooperative matrix (lib/Runtime/VCRuntime.cpp)
- Query `VkPhysicalDeviceCooperativeMatrixFeaturesKHR` via Features2 pNext chain.
- If supported: enable it on the device pNext chain + add
  `VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME` to devExts (conditional, like printf).
- Store a `bool cooperativeMatrix` on the device for capability reporting
  (vcGetDeviceProperties could expose it; optional for P3B).
- This is what lets vkCreateDevice accept a shader using OpCooperativeMatrix*.
  NOTE: also need `shaderFloat16` (already advertised in target_env as Float16
  capability; runtime already enables shaderFloat16 when supported per the
  getVCTargetEnv comment — verify VCRuntime enables shaderFloat16, add if missing).

### Step 8 — GLSL backend guard (lib/Codegen/ASTToGLSL.cpp)
- If a kernel references `wmma::`, emit `error: wmma requires MLIR backend
  (-emit=full); GLSL backend does not support cooperative matrix`. Don't lower.

### Step 9 — Demo + tests
- `test/wmma_gemm.vc` (e2e): M=8,N=8,K=16. A (8x16 fp16), B (16x8 fp16), C (8x8
  fp32, zeroed). Kernel: one warp (block=32) cooperatively holds the tile. load A,
  B, C; mma_sync(D,A,B,C); store D. Host: fill A/B with known fp16 values, C=0,
  launch, readback D (8x8 fp32), verify D==A*B in fp32. Use the e2e harness pattern
  (vcMalloc + vcMemcpy + launch + vcMemcpy back + printf PASS/FAIL).
  - GOTCHA: only ONE warp does the MMA; block size MUST be 32 (the warp). The MLIR
    host e2e path hardcodes 32 anyway. Threads 0..31 cooperatively hold the tile;
    do not guard threads out of the warp.
  - Strides: A row-major ldm=16 (K, in fp16 elements), B col-major ldm=16 (K), C
    row-major ldm=8 (N). Stride is in ELEMENTS per the SPIR-V spec.
  - fp16 host values: use `_Float16` or manual bit conversion in the demo's main().
- `test/MLIR/wmma.mlir` (lit, IR check): RUN mlirc -emit=mlir, FileCheck for
  `spirv.coopmatrix`, `spirv.KHR.CooperativeMatrixLoad/Store/MulAdd`, and the
  target_env V_1_6 + CooperativeMatrixKHR in the gpu.module attr.
- `test/Frontend/wmma.vc` (lit, sema): -fsyntax-only, check fragment decl +
  call arg type errors (e.g. mma_sync with non-accumulator d, load with wrong
  arity, store of a non-accumulator fragment).

## Open risks / to verify during impl
- **GPU shape CONFIRMED**: this dev GPU supports 8x8x16 fp16→fp32 (A/B=FLOAT16,
  C/D=FLOAT32, scope=Subgroup), compute stage. 16x16x16 NOT supported — API is
  Vulkan-native shape (8x8x16 for the demo). e2e CAN run on this GPU.
- **spirv::CooperativeMatrixUseKHR enum name** in the C++ API — assembly keyword is
  `MatrixAcc`; verify the C++ enum value name when writing step 5.
- **GPUToSPIRV legalization of spirv.KHR.CooperativeMatrix***: these are already
  spirv dialect ops (not gpu/memref), so they pass through GPUToSPIRV unchanged
  (it only lowers gpu.*/memref.*/arith etc.). The serializer (SPIRVBinaryUtils)
  must know the opcodes — confirm the KHR ops are registered in the built MLIR.
  Risk: the VC-linked MLIR might not include the CooperativeMatrix ops if built
  with a subset. Check `build-mlir` MLIR has spirv.KHR.CooperativeMatrixLoad
  registered (mlir-tblgen generated them into SPIRVOps.cpp.inc per the research).
- **block size 32 hardcoded** in MLIR host e2e path — fine for WMMA (warp=32).
- **fp16 SSBO layout**: A=8x16 fp16=256B, B=16x8 fp16=256B, C/D=8x8 fp32=256B.
  Cast the memref pointer to `!spirv.ptr<f16, StorageBuffer>` (or f32 for C).
  stride in ELEMENTS (A row-major ldm=16, B col-major ldm=16, C row-major ldm=8).

## Verification gate
1. `cmake --build build-mlir --target vc vcc` compiles.
2. lit: `test/MLIR/wmma.mlir` + `test/Frontend/wmma.vc` pass.
3. `test/wmma_gemm.vc` e2e PASS (vc -emit=full → run → D==A*B). GPU supports
   8x8x16 fp16→fp32, so e2e runs on this machine.
4. No regression: existing lit + e2e still pass (96/97 baseline; only pre-existing
   printf.vc::glsl FAIL).

## Status — DONE (2026-09-17)

P3B is complete and verified end-to-end. `wmma_gemm: PASS` on the MLIR backend;
98/98 (backend,demo) e2e pairs pass (2 skipped: printf GLSL-only, wmma_gemm
GLSL-rejected). The final implementation diverged from the original plan above
in several places; this section records what was actually built.

### Lowering: gpu.subgroup_mma (NOT hand-emitted spirv.KHR)

The plan sketched hand-emitting `spirv.Variable`/`spirv.AccessChain`/
`spirv.KHR.CooperativeMatrix*` directly. That does not work: those spirv ops
are illegal in a func.func body *before* GPUToSPIRV (the pass expects
gpu/memref/arith/scf and legalizes them). The working path is MLIR's idiomatic
GPU-dialect WMMA ops, which the built-in `populateGpuWMMAToSPIRVCoopMatrixKHRConversionPatterns`
+ `populateMMAToSPIRVCoopMatrixTypeConversion` (already bundled in
`createConvertGPUToSPIRVPass`) lower to `spirv.KHR.CooperativeMatrix*`:

- `wmma::fragment<use,M,N,K,T,layout>` → `!gpu.mma_matrix<MxNxT, operand>` where
  operand ∈ {"AOp","BOp","COp"} and the shape is the per-operand tile
  (A→M×K, B→K×N, C→M×N). `cvtType` builds this in ASTToMLIR.
- `load_matrix_sync` → `gpu.subgroup_mma_load_matrix` (memref operand, `[c0,c0]`
  offset, `leadDimension` attr, optional `transpose` unit attr).
- `mma_sync` → `gpu.subgroup_mma_compute` (a, b, c → result; a_transpose/
  b_transpose unit attrs present but unused).
- `store_matrix_sync` → `gpu.subgroup_mma_store_matrix`.

Fragments are register-resident SSA values (no storage allocation); a
`wmmaFragments` StringMap<Value> holds the current matrix value per name.

### transpose flag = "source is col_major" (uniform, NOT the CUDA-convention split)

`gpu.subgroup_mma_load_matrix`'s `transpose` flag uniformly selects the SPIR-V
`CooperativeMatrixLayoutKHR`: transpose=false → RowMajor, transpose=true →
ColumnMajor, for ALL operands (A, B, C). So `fragTransposed` is simply
`layout == ColMajor`. This differs from the CUDA WMMA textbook convention
(where matrix_b's natural layout is col-major and row-major B "transposes");
MLIR's lowering does not mirror that asymmetry. The wmma_gemm demo (A, B, C all
row-major in memory) loads all three with transpose=false → RowMajor, and
`D = A×B + C` verifies correct.

### Three post-conversion fixups in LoweringPasses.cpp

The built-in GPUToSPIRV + spirv-lower-abi-attrs pipeline leaves three things
broken for the 1.6 cooperative-matrix target; each is fixed by a small
post-pass in `runLoweringPipeline`:

1. **`rewriteMarkedAtomics`** (pre-existing): GPUToSPIRV has no lowering for
   `memref.atomic_rmw` 'assign' (atomicExch); ASTToMLIR emits it as a marked
   AtomicIAdd, matched by ordinal position post-conversion and rewritten to
   spirv.AtomicExchange/AtomicXor.
2. **`fixupF16StorageBuffers`** (new): MLIR's MemRef→SPIRV storage-buffer type
   converter widens every sub-4-byte SSBO element to 4 bytes (f16→f32, stride=4).
   A `spirv.KHR.CooperativeMatrixLoad` takes the SSBO pointer directly, so it
   would read 4-byte elements into an f16 matrix (reinterpreting each float as
   two halfs) — silently wrong (spirv-val does not catch it). This pass traces
   each f16 cooperative-matrix load/store back through AccessChain→addressof→
   global, and rebuilds that global's chain with `rtarray<f16, stride=2>` /
   `ptr<f16>` so the cooperative matrix reads true 16-bit halfs. Scoped to the
   cooperative-matrix path only (scalar f16/i8 SSBO access still hits the
   generic widening — a separate, larger fix).
3. **`populateEntryPointInterfaces`** (new): SPIR-V 1.4+ requires every
   statically-used interface variable (INCLUDING StorageBuffer descriptor
   variables) to be listed in OpEntryPoint's interface list. The cooperative-
   matrix target is 1.6, but spirv-lower-abi-attrs only lists the Input/Output
   builtins it creates (WorkgroupId, LocalInvocationId, ...), not the SSBO
   globals → spirv-val "interface variable used but not listed". This pass walks
   each spirv.func, collects descriptor globals referenced via
   spirv.mlir.addressof, and merges them into the matching EntryPoint's
   interface attr.

### Runtime: opportunistic cooperative-matrix enablement

VCRuntime device creation queries `VkPhysicalDeviceCooperativeMatrixFeaturesKHR`
+ `shaderFloat16`; if both are supported, enables them and pulls in
`VK_KHR_cooperative_matrix` (device ext) + bumps the SPIR-V target to 1.6.
`VulkanDevice::coopMatrix` flags support. Kernels that don't use wmma are
unaffected. The Intel ARL dev GPU reports both features and the 8x8x16
fp16→fp32 Subgroup shape.

### Host backend fixes (general, surfaced by wmma_gemm)

Two pre-existing host-codegen bugs surfaced (wmma_gemm is the first demo with
`__half` host arrays sized by `const int`):

- **`const int` array dimensions**: `T arr[M*K]` where M,K are `const int`
  locals produced `T arr[0]` (the parser's const-folder didn't resolve
  DeclRefExpr to const variables) → stack smash → `getenv` crash in the Intel
  driver during beginFrame. Fixed by tracking `const int NAME = <literal>`
  values in the parser (`constInts` map) and resolving DeclRefExpr in
  `evalConstInt`.
- **`__half` host type**: the host backend degraded `__half` to `float` (4
  bytes), so a `__half arr[N]` occupied 4N bytes while `vcMemcpy(2N)` copied
  only half — corrupting the byte stream the device read as packed f16. Fixed
  by emitting `_Float16` (GNU/C11 16-bit float) for `BuiltinTypeKind::Float16`
  on the host.

### GLSL backend guard

`vcc` rejects `wmma::` with a diagnostic ("tensor-core intrinsics require the
MLIR backend"), via `Sema::usesCoopMatrix()`. The e2e harness skips
wmma_gemm on the GLSL backend (`GLSL_UNSUPPORTED` in run_e2e.py) so the suite
reports it as skipped, not a build failure.

