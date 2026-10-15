#!/usr/bin/env python3
"""Emit an nvidia-pcc pipeline JSON for a VC-generated compute kernel.

nvidia-pcc (NVIDIA's Vulkan SC pipeline compiler) does not read SPIR-V
directly: it takes a JSON *descriptor* of the pipeline plus the .spv file it
names.  The JSON is a normal Vulkan pipeline description in NVIDIA's own
schema — same shapes as the vendor sample under
/ota/samples/vulkan-sc-samples/data/pipeline/<sample>/, so we start from that
sample's JSON as a template and rewrite only what a VC kernel changes:

  * ShaderFileNames[0].filename -> the VC kernel's .spv
  * the descriptor-set-layout bindings -> one STORAGE_BUFFER per kernel
    SSBO/image parameter, in VC's binding order
  * the push-constant range -> the kernel's scalar-argument block
  * PipelineUUID -> a UUID we choose, so we can find the pipeline again
    through VkPipelineOfflineCreateInfo at runtime

Usage:
  gen_pcc_json.py --template <vendor .json> --spv vk_vadd_pipeline_1.compute.spv
                  --uuid 0102030405060708090a0b0c0d0e0f10
                  --bindings storage,storage,storage
                  --push-constant offset=0,size=4
                  -o pipeline/vk_vadd_pipeline_1.json

Then, on an x86-64 host with the driver .run unpacked:
  nvidia-pcc -path <json dir> -prefix <name> -chip gb10b -out pipeline_cache.bin
and patch the 32-byte cache header (see patch_cache_header.py) before the
device will accept it.
"""
import argparse
import json
import os
import sys

# VkDescriptorType values, as spelled in the pcc JSON schema.
DESC_TYPES = {
    "storage": "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER",
    "uniform": "VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER",
}
STAGE_COMPUTE = "VK_SHADER_STAGE_COMPUTE_BIT"


def build(template_path, spv_basename, uuid_hex, bindings, push_constant):
    with open(template_path) as f:
        doc = json.load(f)

    state = doc["ComputePipelineState"]

    # --- descriptor set layout: rebuild the binding list from scratch --------
    set0 = state["DescriptorSetLayouts"][0]["pSetLayouts_0"]
    set0["pBindings"] = [
        {
            "binding": i,
            "descriptorType": DESC_TYPES[kind],
            "descriptorCount": 1,
            "stageFlags": STAGE_COMPUTE,
            "pImmutableSamplers": "NULL",
        }
        for i, kind in enumerate(bindings)
    ]
    set0["bindingCount"] = len(bindings)

    # --- pipeline layout: at most one push-constant range --------------------
    layout = state["PipelineLayout"]
    if push_constant:
        off, size = push_constant
        layout["pushConstantRangeCount"] = 1
        layout["pPushConstantRanges"] = [
            {
                "sType": "VK_STRUCTURE_TYPE_PUSH_CONSTANT_RANGE",
                "stageFlags": STAGE_COMPUTE,
                "offset": off,
                "size": size,
            }
        ]
    else:
        layout["pushConstantRangeCount"] = 0
        layout["pPushConstantRanges"] = []

    # --- the SPIR-V module pcc should compile -------------------------------
    state["ShaderFileNames"][0]["filename"] = spv_basename

    # --- identity: pcc copies this UUID into the cache; the app looks the
    # pipeline up again with VkPipelineOfflineCreateInfo.pipelineIdentifier --
    doc["PipelineUUID"] = list(bytes.fromhex(uuid_hex))

    return doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True,
                    help="vendor sample pipeline JSON to use as the schema template")
    ap.add_argument("--spv", required=True,
                    help="basename of the .spv that will sit next to the JSON")
    ap.add_argument("--uuid", required=True, help="32 hex chars")
    ap.add_argument("--bindings", default="",
                    help="comma-separated descriptor kinds, e.g. storage,storage,storage")
    ap.add_argument("--push-constant", default="",
                    help="offset=N,size=N (omit for no push constants)")
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    if len(args.uuid) != 32:
        sys.exit("--uuid must be 32 hex chars")
    bindings = [b for b in args.bindings.split(",") if b]
    pc = None
    if args.push_constant:
        kv = dict(p.split("=") for p in args.push_constant.split(","))
        pc = (int(kv["offset"]), int(kv["size"]))

    doc = build(args.template, args.spv, args.uuid, bindings, pc)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(doc, f, indent=4)
        f.write("\n")
    print("%s: %d bindings, push_constant=%s, uuid=%s"
          % (args.output, len(bindings), pc, args.uuid))


if __name__ == "__main__":
    main()
