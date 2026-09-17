# P3C — scalar __half* SSBO load/store (MLIR backend)

## Goal
Make scalar `__half*` SSBO load/store work end-to-end on the MLIR backend.
P3B left this as a documented known limitation: `fixupF16StorageBuffers`
only handled the cooperative-matrix (KHR) path, not scalar `spirv.Load`/
`spirv.Store`. `test/half.vc` even dodged the bug by reading/writing a
`float*` SSBO and keeping half arithmetic internal.

## Scope (confirmed by probing)
- **Only f16 SSBO is affected.** VC has NO i8/i16 types (BuiltinTypeKind:
  Int32/UInt32/Int64/UInt64 + Float16/32/64). `char`/`short` map to Int32
  or are rejected before reaching the SSBO converter. No i8 widening to fix.
- **`__shared__` (Workgroup) memory NOT affected** — widening is
  StorageBuffer-only. `__shared__ __half s[8]` lowers correctly.
- **`__half2` unsupported** (ASTToMLIR rejects local half2) — out of scope.
- Coop-matrix path (wmma_gemm) UNAFFECTED by this bug; the extended
  fixupF16StorageBuffers handles both KHR and scalar paths disjointly.

## Root cause (CONFIRMED by direct IR dump)
MLIR 18 `SPIRVTypeConverter` widens sub-4-byte StorageBuffer elements to
4 bytes: `memref<?xf16, StorageBuffer>` → `ptr<struct<(rtarray<f32,stride=4>)>, StorageBuffer>`.

The MemRef→SPIRV **load** pattern leaves an unresolved bridge cast:
```
%15 = spirv.Load "StorageBuffer" %14 : f32
%16 = builtin.unrealized_conversion_cast %15 : f32 to f16   // residue
%17 = spirv.FConvert %16 : f16 to f32
```
The **store** pattern is type-mismatched:
```
%22 = spirv.FConvert %21 : f32 to f16
spirv.Store "StorageBuffer" %27, %22 : f16   // MISMATCH: ptr<f32> vs f16
```
Verification of this store fails → `pm.run` returns failure → the
spirv.module passes that run AFTER conversion (`spirv-lower-abi-attrs` +
`SPIRVUpdateVCEPass`) never run → the spirv.module has no EntryPoint and
no `vce_triple`.

## Key finding: conversion does NOT destroy the module
After a failed `pm.run`, the `spirv.module` survives fully populated with
exactly the widening residues + the missing vce_triple. Because
spirv-lower-abi-attrs never ran, the SSBO base in the scalar path is a
**spirv.func BlockArgument** (args not yet converted to globals), whereas
the coop-matrix path (conversion succeeds) has **AddressOfOp→GlobalVariable**
bases. `fixupF16StorageBuffers` handles BOTH base forms.

The widened `rtarray<f32,stride=4>` is a genuine LAYOUT bug (host
`vcMalloc`'s packed f16, 2 bytes/elem), not just a type inconvenience →
must NARROW to `rtarray<f16, stride=2>`, not repair the widened form.

## Approach: extend fixupF16StorageBuffers (post-conversion narrowing)

`fixupF16StorageBuffers` is extended from a KHR-only fixup to a 4-pass
post-conversion repair covering BOTH coop-matrix and scalar f16 SSBO
access. Run after `rewriteMarkedAtomics`, before
`rerunAbortedSPIRVLegality` + `populateEntryPointInterfaces`.

### Pass 1 — collect f16 SSBO bases
Walk `spirv::AccessChainOp`s; for each, inspect its users to decide
whether this access chain reaches an f16 element:
- `KHRCooperativeMatrixLoadOp` / `KHRCooperativeMatrixStoreOp` →
  coopmatrix element type.
- `spirv::StoreOp` → `store.getValue().getType()` (scalar store value).
- `spirv::LoadOp` → look for an `unrealized_conversion_cast` f32→f16
  user (the bridge cast the load pattern left); if found, elemTy = f16.
- **Guard the `elemTy.isF16()` check on `elemTy` being non-null** — the
  LoadOp branch leaves `mlir::Type elemTy;` default-constructed (null)
  when no bridge cast is found, and `isF16()` on a null Type dereferences
  a null impl pointer → segfault on any non-f16 AccessChain.
Record the base Value (`ac.getBasePtr()`) into `f16Bases`.

### Pass 2 — narrow each f16 base + rebuild AccessChains
Narrowed types: `rtarray<f16,stride=2>` in a struct at offset 0,
`ptr<struct<...>, StorageBuffer>` (base) and `ptr<f16, StorageBuffer>` (elem).
- **BlockArgument base** (scalar path, pre-lower-abi-attrs): mutate the
  arg type to `ptrTy`, rebuild every AccessChain on it to yield
  `ptr<f16>`, RAUW + erase old chains.
- **AddressOfOp base** (coop-matrix path, post-lower-abi-attrs): narrow
  the referenced `spirv.GlobalVariable`'s `type` attr, rebuild the
  addressof + its AccessChains, RAUW + erase.

### Pass 3 — rebuild scalar Loads to yield f16
After narrowing, a scalar `spirv.Load` on an f16 SSBO still yields f32
(the widened result baked in by the pattern) while its pointer is now
`ptr<f16>` — a mismatch. Rebuild each such Load (`ptr<f16>` + f32 result)
to yield f16, RAUW the bridge-cast users to the new load, erase the old
load + cast. Scalar Store is already fixed (pointer is `ptr<f16>`,
value was f16 all along).

### Pass 4 — augment the spirv.target_env
Narrowing the SSBO to `ptr<f16>` produces 16-bit storage buffer access
(`spirv.Bitcast` over `ptr<f16>`, f16 loads/stores into a StorageBuffer),
which requires the `StorageBuffer16BitAccess` capability +
`SPV_KHR_16bit_storage` extension and SPIR-V >= 1.3. The original
target_env only advertised `Float16` (f16 arithmetic), not 16-bit
*storage*. Augment the spirv.module's `spirv.target_env` (bump to 1.3,
add the capability + extension) so the re-run `SPIRVUpdateVCEPass`
stamps a `vce_triple` that admits the 16-bit-storage ops; otherwise
`spirv-lower-abi-attrs` rejects the `spirv.Bitcast` ("requires
SPV_KHR_16bit_storage"). Done only when an f16 SSBO was narrowed.

## rerunAbortedSPIRVLegality
The in-pipeline `pm.run` aborts on the pre-narrowing store mismatch, so
`spirv-lower-abi-attrs` (EntryPoint + abi globals) and
`SPIRVUpdateVCEPass` (vce_triple) never ran. After `fixupF16StorageBuffers`
makes the IR type-consistent, re-run those two passes on a fresh
`PassManager` to finish legalization and make the module serializable.

The in-pipeline `pm.run` failure is EXPECTED and recovered for f16-SSBO
shaders; `runLoweringPipeline` no longer `emitError`s on that failure
(it would noise up stderr despite a successful build). A genuine non-f16
lowering failure leaves the spirv.module unserializable, caught by
`rerunAbortedSPIRVLegality`'s `emitError` or the driver's
"spirv translation failed".

## populateEntryPointInterfaces
SPIR-V < 1.4: listing StorageBuffer vars in OpEntryPoint's interface is
ILLEGAL ("interfaces must be OpVariables with Storage Class Input(1) or
Output(3). Found Storage Class 12"). Guard with
`vce->getVersion() >= V_1_4` (the f16-SSBO target is bumped to 1.3, so
this stays a no-op for scalar f16 SSBO; the coop-matrix target is 1.6).

## Runtime: opportunistic 16-bit storage feature enablement
The device must enable `shaderFloat16` (f16 arithmetic) AND
`storageBuffer16BitAccess` (16-bit values in SSBOs) for the narrowed
shaders to validate. Added to `VCRuntime` device creation:
- Query `VkPhysicalDevice16BitStorageFeatures` (chained off the Vulkan12
  query) for `storageBuffer16BitAccess`.
- When the device supports both `shaderFloat16` AND
  `storageBuffer16BitAccess`, enable them (composing with the existing
  coopMatrix `shaderFloat16` enablement) and thread the
  `16BitStorageFeatures` struct into the device-create pNext chain.
- New `VCDevice::f16Storage` flag gates the chain threading.
Enabled opportunistically — kernels that don't use `__half*` SSBOs are
unaffected.

## Residual known noise
`vc -emit=spirv|full` on an f16-SSBO shader prints one stderr line:
`error: 'spirv.Store' op mismatch in result type and pointer type`. This
is MLIR's op verifier reporting the very store mismatch the conversion
pattern emits DURING `pm.run`, before `fixupF16StorageBuffers` repairs
it. The build succeeds (exit 0), spirv-val passes, and the shader runs
correctly. Suppressing it would require either a scoped diagnostic
filter (risks hiding real errors) or a pre-conversion rewrite (rejected:
conflicts with the coop-matrix contiguous-f16 tile layout on a shared
global). Left as documented noise.

## Why NOT a pre-conversion rewrite
- An i32 backing view / scaled index breaks coop-matrix contiguous f16
  tile layout — conflicts with the KHR path on a shared global.
- The converter widens the memref TYPE itself; we cannot force f16
  through a widened global via pre-pass load/store rewriting.
- The `__shared__` model (direct GlobalVariable+Load/Store at f16)
  doesn't transfer: SSBO globals are kernel args created by GPUToSPIRV
  during conversion, not module-scope pre-conversion.
- Post-conversion is the minimal, proven approach (P3B precedent) and
  the module survives the failed conversion.

## Verification (ALL PASS)
1. `test/f16_ssbo.vc` — new e2e demo: scalar `__half*` SSBO load,
   scale by 2, store. PASSES on MLIR backend (0,2,4,...,14).
2. `test/f16_ssbo.vc` PASSES on GLSL backend too (handles
   `__half`→float16_t scalar SSBO already).
3. `wmma_gemm.vc` still PASS (coop-matrix path untouched).
4. Full e2e suite: 100/100 pairs passed, 2 skipped (no regression;
   was 98/98 + 2 skipped, +f16_ssbo×2 backends).
5. `spirv-val` on the produced `.spv` passes.
6. Frontend (lit) tests pass.

## Files
- `lib/Codegen/LoweringPasses.cpp` — extend `fixupF16StorageBuffers`
  (4 passes: collect/narrow/rebuild-loads/augment-target-env), add
  `rerunAbortedSPIRVLegality`, version-guard `populateEntryPointInterfaces`,
  suppress the in-pipeline `emitError`; wire into `runLoweringPipeline`.
- `lib/Codegen/VCToGPU.cpp` — (unchanged; target_env augmentation is done
  in the fixup pass where narrowing is known to be needed).
- `lib/Runtime/RuntimeInternal.h` — add `VCDevice::f16Storage` flag.
- `lib/Runtime/VCRuntime.cpp` — opportunistic `shaderFloat16` +
  `storageBuffer16BitAccess` enablement + pNext chain threading.
- `test/f16_ssbo.vc` — new e2e demo.
- `test/run_e2e.py` — (unchanged; f16_ssbo needs no skip on either
  backend).
- `PLAN_f16_ssbo.md` — this plan.
