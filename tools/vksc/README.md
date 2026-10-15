# Vulkan SC / nvidia-pcc tooling

Helpers for compiling VC-generated SPIR-V into a Vulkan SC
`pipeline_cache.bin`, and for probing / debugging the SC driver's offline
pipeline path.  Everything here is host-side glue — it does not link against
VC and is not part of the runtime.

Target used for all of the numbers below: **NVIDIA Thor GB10B**, Vulkan SC
driver **581.1.0** (`/usr/lib/libnvidia-vksc-core.so.1`), vendor samples at
`/ota/samples/vulkan-sc-samples`.

## Why this exists

Vulkan SC removed runtime pipeline compilation: pipelines must be compiled
*ahead of time* by NVIDIA's `nvidia-pcc` and shipped inside a
`pipeline_cache.bin`.  The runtime then creates them with an *offline*
create-info (`VkPipelineOfflineCreateInfo`), selecting the entry by 16-byte
UUID instead of by shader module.

`nvidia-pcc` does not consume SPIR-V directly — it takes a JSON descriptor of
the pipeline plus the `.spv` file that descriptor names, so feeding VC's
output to it requires generating that JSON.

## Pipeline

```
vcc kernel.vc -emit=spirv -o vk_x_pipeline_1.compute.spv
gen_pcc_json.py --template <vendor sample json> --spv vk_x_pipeline_1.compute.spv \
    --uuid 0102... --bindings storage,storage,storage --push-constant offset=0,size=4 \
    -o pipeline/vk_x_pipeline_1.json
nvidia-pcc -path pipeline -prefix vk_x_pipeline_1 -chip gb10b -out cache.bin
patch_cache_header.py --in cache.bin --out cache_devid0.bin
scpipe_probe.c  (on the device)  check the pipeline can actually be created
```

`nvidia-pcc` runs on **x86-64 only** and ships inside the plain NVIDIA driver
`.run` (e.g. `NVIDIA-Linux-x86_64-595.99.02`), *not* in the SC package and
*not* on the device.  Invoke it as:

```
LD_LIBRARY_PATH=<driver dir> <driver dir>/nvidia-pcc \
    -path <json dir> -prefix <name> -chip gb10b -out <cache.bin>
```

`-chip gb10b` is Thor's chip name; `nvidia-pcc --help` lists the alternatives.

**`-prefix` is a *filename* filter, not a name for the output.**  `pcc` scans
the `-path` directory for files whose name starts with the prefix and compiles
exactly those.  Feeding it a partial set (`-prefix p_` when the directory holds
`q_*.json` too) produces a cache that is silently short: the pipelines it did
not see fail much later at `vkCreateComputePipelines` with
`VK_ERROR_NO_PIPELINE_MATCH`, i.e. it looks like a UUID mismatch rather than a
missing compile.  Always compile **every** pipeline in one run, under one
common prefix.  `build_cache.sh` stages into a fresh directory so this cannot
happen.

### Two header fields must be patched

`nvidia-pcc` cannot know the target's identity, so two fields of the cache
header come out wrong and the SC device rejects the whole cache with
`VK_ERROR_VALIDATION_FAILED` (-1000011001) — *before* it ever looks at the
UUID, which makes this look like a pipeline problem when it is not:

| offset | field | must be |
|---|---|---|
| `0x0c` | `deviceID` | exactly `0` |
| `0x10..0x20` | `pipelineCacheUUID` | device's `VkPhysicalDeviceProperties::pipelineCacheUUID` (Thor GB10B/581: `e251a40589539d030000000000000000`) |

`patch_cache_header.py` does both; `--show` dumps the header.

## The end-to-end flow: `vc -emit=sc` -> `build_cache.sh` -> run

The manual pipeline above is what the runtime's own flow automates.  `vc
-emit=sc <file.vc> -o <dir>` writes, for every `__global__` kernel **and every
distinct launch-site block size**:

```
<dir>/<kernel>.b<X>_<Y>_<Z>.spv    block-size-baked SPIR-V
<dir>/<kernel>.b<X>_<Y>_<Z>.json   pcc descriptor
<dir>/<stem>_host.cpp              host code calling vcLoadKernelOffline
```

then `tools/vksc/build_cache.sh <dir>` runs one `nvidia-pcc` over the whole
directory and patches the header, leaving `<dir>/pipeline_cache.bin`:

```
vc -emit=sc examples/sc_vadd.vc -o build/sc
tools/vksc/build_cache.sh build/sc            # -> build/sc/pipeline_cache.bin
<aarch64-g++> build/sc/sc_vadd_host.cpp -o sc_vadd -lVCRuntime -lvulkansc
scp sc_vadd build/sc/pipeline_cache.bin <target>:~ ; run it there
```

Two properties of the flow are worth knowing:

* **The block size is baked into the binary**, not passed as a specialization
  constant.  SC validates the app's create-info against the cached entry, and a
  `VkSpecializationInfo` that pcc compiled away is exactly the kind of mismatch
  that returns a bare `-1000011001`.  `vc -emit=sc` writes a literal
  `OpExecutionMode LocalSize` instead (one binary per block size) and the
  runtime passes `pSpecializationInfo = NULL`.  A launch whose block size is not
  a compile-time constant is a hard error in `sc` mode.
* **The compiler owns the UUID.**  The same table produces the pcc JSON, the
  `VCScPipelineDesc` in the host code, and the create-info's
  `pipelineIdentifier`, so they cannot drift — a drift shows up only as
  `VK_ERROR_NO_PIPELINE_MATCH` at the first dispatch.  The UUID is a hash of the
  *content* (the serialized SPIR-V plus entry point and block dims), not of the
  kernel name: two files that each define a kernel called `vadd` are two
  different pipelines and must not share an identifier, and moving a `.vc` file
  must not invalidate an already-built cache.
* **A handle carries its block size.**  `__vc_k_<kernel>_b<x>_<y>_<z>` is one
  per (kernel, block size) — dispatching the wrong one is prevented by the
  handle, not by a runtime check.  Host code that names a handle explicitly
  (the `vcLaunchKernelIndirect(__vc_k_f, ...)` / `vcOccupancyMaxPotentialBlockSize(...,
  __vc_k_f, ...)` idiom the default backends support) gets a reference alias
  `__vc_k_<kernel>` to that kernel's first block size; a named use at any other
  block size is refused at dispatch with both sizes named.
* **Images are bound after the buffers.**  A kernel taking both an SSBO and a
  storage image gets bindings `0..N-1` (SSBOs) then `N..N+M-1` (images), and
  those are absolute binding numbers in both the runtime's layout and the pcc
  JSON.  Numbering the image sequence from zero instead collides with the first
  SSBO, and this profile *accepts* the resulting layout — the kernel then
  dispatches with the image where the buffer should be and every store vanishes.

## What the Thor SC profile does (and does not) allow

The SC profile is a subset of Vulkan, and which parts it carries is a *device
property*, not an assumption.  GB10B / driver 581.1.0 reports:

| `VkPhysicalDeviceVulkanSC10Properties` | value | consequence in the runtime |
|---|---|---|
| `commandPoolResetCommandBuffer` | **`VK_FALSE`** | `vkResetCommandBuffer` is rejected, see below |
| `commandPoolMultipleCommandBuffersRecording` | `VK_FALSE` | one recording command buffer per pool |
| `recycleDescriptorSetMemory` | **`VK_FALSE`** | `vkResetDescriptorPool` gated off; pools are allocated to exhaustion |
| `recyclePipelineMemory` | `VK_TRUE` | pipelines may be destroyed/reused |
| `deviceDestroyFreesMemory` | **`VK_FALSE`** | never `vkFreeMemory`; `vkDestroyDevice` does not reclaim either |
| `deviceNoDynamicHostAllocations` | `VK_TRUE` | every object must be in `VkDeviceObjectReservationCreateInfo` |

`VC_SC_TRACE=1` prints this line at `vcInit` and then traces every frame
boundary's `VkResult`, which is how the first three were found.

**`commandPoolResetCommandBuffer == VK_FALSE` is the expensive one.**  On this
profile `vkResetCommandBuffer` on a command buffer that has already been
submitted returns `VK_ERROR_VALIDATION_FAILED` (-1000011001) — and, worse, every
command recorded into that buffer *afterwards* is dropped **without any error
from `vkBeginCommandBuffer`, `vkCmdCopyBuffer` or `vkQueueSubmit`**.  The frame
submits empty command buffers forever: kernels and copies silently do nothing
while every runtime call returns success.  The runtime therefore hands each
frame a freshly allocated command buffer under this profile (freeing the
submitted one returns its memory to the pool reservation, so the peak stays at
one command buffer per frame).  Symptom to recognise: the first submission on
each frame slot works, everything after it is a no-op.

### Reservation overruns masquerade as OOM

Exceeding a reserved object count does **not** fail at `vkCreateDevice` — it
fails at the first create of that object type, with
`VK_ERROR_OUT_OF_HOST_MEMORY` (-1).  Known instances: `pipelineCacheRequestCount`
(→ `vkCreatePipelineCache`), `computePipelineRequestCount` / the pipeline pool
`poolEntryCount` (→ `vkCreateComputePipelines`), and
`descriptorSetLayoutBindingRequestCount` (→ `vkCreateDescriptorSetLayout`).
Every default is generous, but an app that pins the reservation can trip it.

## Debugging an offline create

`VK_ERROR_VALIDATION_FAILED` on `vkCreateComputePipelines` is unhelpful
because it covers a lot of ground.  Two hard-won facts:

1. **The app-supplied create info is validated against the cached entry, so
   the pipeline layout must match the JSON exactly** — same descriptor-set
   layout bindings (type *and* count), same push-constant ranges.  An empty
   layout that "would normally be fine" is rejected outright.
2. **`descriptorSetLayoutBindingLimit` in `VkDeviceObjectReservationCreateInfo`
   is a real per-layout cap.**  The vendor sample sets it to 2 for its
   2-binding layout; a 3-binding layout needs it raised to 3, otherwise
   `vkCreateDescriptorSetLayout` itself fails with -1000011001.  It is not a
   pool size or a total across layouts.

`vk_create_tap.c` is an `LD_PRELOAD` shim that dumps the real create-info
structs — `vkCreateDevice` (incl. the whole object reservation),
`vkCreatePipelineCache` / `DescriptorSetLayout` / `PipelineLayout` /
`ComputePipelines` — and forwards to the driver.  It resolves the real entry
points via `dlsym(RTLD_NEXT, ...)`, since most apps link `libvulkansc.so.1`
directly and never go through a loader.  Use it to capture what a known-good
app actually passes:

```
aarch64-linux-gnu-gcc -O1 -shared -fPIC vk_create_tap.c -o vk_create_tap.so -ldl
LD_PRELOAD=./vk_create_tap.so <app>          # see run_tap.sh
```

`scpipe_probe.c` is the standalone probe: it creates the SC device with the
same reservation counts, builds the layout described by `--bindings` /
push-constant, and tries the offline create.  Cross-compile with:

```
aarch64-linux-gnu-gcc -O1 -I<vksc headers> scpipe_probe.c -o scpipe_probe -ldl
./scpipe_probe <cache.bin> <uuid-hex32> <vadd|vendor> [driver.so]
```

`vadd` mode = 3 STORAGE_BUFFER bindings + push constant `{offset=0,size=4}`
(matching `vk_vadd_pipeline_1.json`); `vendor` mode reproduces the
computeparticles sample's 2-binding layout, which is a useful control: the
vendor's own shipped cache must pass through it.

## Verified state

The vendor sample's own cache and a VC-built cache both create successfully
through `scpipe_probe`:

```
$ ./scpipe_probe <vendor cache.bin> d0ffd0ff80ff80ffa0ffe0ff80ff20ff vendor
createPipelineCache(from pcc data)=0
createDescriptorSetLayout(bindings=2)=0
createPipelineLayout(pc=0)=0
createComputePipelines(offline, cache UUID)=0   <- PIPELINE CREATED

$ ./scpipe_probe cache_devid0.bin 0102030405060708090a0b0c0d0e0f10 vadd
createPipelineCache(from pcc data)=0
createDescriptorSetLayout(bindings=3)=0
createPipelineLayout(pc=1)=0
createComputePipelines(offline, cache UUID)=0   <- PIPELINE CREATED
```

i.e. **unmodified VC-emitted SPIR-V**, wrapped in a pcc-built cache, is
accepted by the SC driver on Thor and resolves to a real `VkPipeline`.

Since then the whole runtime runs that way: a VC program compiled with
`vc -emit=sc`, its cache built by `build_cache.sh`, and driven by the ordinary
`vcMalloc` / `vcMemcpy` / `vcLaunchKernel` / `vcDeviceSynchronize` API runs on
the Thor SC driver —

```
$ ./sc_vadd                       # both block sizes: <<<N,32>>> and <<<N,64>>>
sc_vadd: PASS
```

A batch of 13 `test/*.vc` demos (atomics, bool, dyn_shared, f16_ssbo, features,
graph, half, image_mixed_ssbo, image_stencil, indirect, managed, struct,
vectors) — 16 pipelines in one cache, exercised through the same runtime — also
passes on the device, which is what covers the parts a single-kernel demo cannot:
many pipelines in one pool, more than one descriptor-set layout, a graph's
secondary command buffer, and a kernel that takes an SSBO *and* a storage image.

with the two failure modes named rather than bare: an under-sized reservation
reports the overrun and which count to raise, and a UUID with no cache entry
reports `no offline pipeline in 'pipeline_cache.bin' matches UUID <hex>` and
refuses the dispatch (`vcLaunchKernel` returns `InvalidKernel`).

## Files

| file | role |
|---|---|
| `gen_pcc_json.py` | VC SPIR-V + vendor template -> pcc pipeline JSON (`--entry-point` for VC kernel symbols) |
| `build_cache.sh` | stage an `-emit=sc` directory, one `nvidia-pcc` run, patch the header -> `pipeline_cache.bin` |
| `patch_cache_header.py` | fix `deviceID` / `pipelineCacheUUID` in `cache.bin` |
| `scpipe_probe.c` | standalone offline-pipeline-create probe (run on device) |
| `vk_create_tap.c` | LD_PRELOAD create-info dumper (debugging) |
| `run_tap.sh` | run the vendor sample under the tap on the Thor box |
| `gt_vendor.sh` | run the vendor sample against a given `pipeline_cache.bin` |
| `vk_vadd_pipeline_1.json` | worked example: VC's `test/vadd.vc` kernel |
