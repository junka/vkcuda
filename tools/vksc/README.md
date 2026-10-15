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

## Files

| file | role |
|---|---|
| `gen_pcc_json.py` | VC SPIR-V + vendor template -> pcc pipeline JSON |
| `patch_cache_header.py` | fix `deviceID` / `pipelineCacheUUID` in `cache.bin` |
| `scpipe_probe.c` | standalone offline-pipeline-create probe (run on device) |
| `vk_create_tap.c` | LD_PRELOAD create-info dumper (debugging) |
| `run_tap.sh` | run the vendor sample under the tap on the Thor box |
| `gt_vendor.sh` | run the vendor sample against a given `pipeline_cache.bin` |
| `vk_vadd_pipeline_1.json` | worked example: VC's `test/vadd.vc` kernel |
