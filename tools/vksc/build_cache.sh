#!/bin/bash
# build_cache.sh — turn a `vc -emit=sc` output directory into the
# pipeline_cache.bin a Vulkan SC runtime loads.
#
#   vc -emit=sc app.vc -o build/sc          # one .spv + .json per (kernel, block)
#   tools/vksc/build_cache.sh build/sc      # -> build/sc/pipeline_cache.bin
#
# Vulkan SC has no runtime pipeline compilation, so every pipeline the app can
# create has to be compiled ahead of time by NVIDIA's nvidia-pcc. pcc takes a
# directory of pipeline JSONs (each naming the .spv it describes) and emits one
# cache holding all of them.
#
# FEED EVERY PIPELINE IN ONE RUN. pcc selects the JSONs by filename prefix and
# each run writes a *fresh* cache, so compiling half of them produces a cache
# whose other lookups fail with VK_ERROR_NO_PIPELINE_MATCH at dispatch time.
# That is why this script stages the whole directory under one common prefix
# instead of looping per kernel.
#
# pcc runs on x86-64 only and ships inside the plain NVIDIA driver .run (e.g.
# NVIDIA-Linux-x86_64-595.99.02), not in the SC package and not on the target.
# Point VC_PCC at it, or put nvidia-pcc on PATH.
#
# Environment:
#   VC_PCC        full path to nvidia-pcc                    (default: PATH)
#   VC_PCC_LIBDIR dir with nvidia-pcc's shared libs         (default: its dir)
#   VC_SC_CHIP    pcc -chip value                           (default: gb10b)
#   VC_SC_UUID    target's pipelineCacheUUID, 32 hex chars  (default: Thor GB10B
#                                                           581 value, see below)
#   VC_SC_OUT     output directory for pipeline_cache.bin   (default: <dir>)
#
# The header still needs two fields fixed afterwards (deviceID=0 and the
# device's pipelineCacheUUID) — pcc cannot know the target's identity. That is
# what patch_cache_header.py does, and the runtime will reject the cache with a
# bare VK_ERROR_VALIDATION_FAILED (-1000011001) without it.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

die() { echo "build_cache: $*" >&2; exit 1; }

srcdir="${1:-}"
[ -n "$srcdir" ] || die "usage: build_cache.sh <dir emitted by 'vc -emit=sc'>"
[ -d "$srcdir" ] || die "not a directory: $srcdir"

shopt -s nullglob
jsons=("$srcdir"/*.json)
spvs=("$srcdir"/*.spv)
[ "${#jsons[@]}" -gt 0 ] || die "no *.json in $srcdir — did 'vc -emit=sc' write here?"
[ "${#spvs[@]}" -gt 0 ] || die "no *.spv in $srcdir — did 'vc -emit=sc' write here?"

chip="${VC_SC_CHIP:-gb10b}"
outdir="${VC_SC_OUT:-$srcdir}"
uuid="${VC_SC_UUID:-e251a40589539d030000000000000000}"

# --- locate nvidia-pcc ------------------------------------------------------
pcc="${VC_PCC:-}"
if [ -z "$pcc" ]; then
  pcc="$(command -v nvidia-pcc || true)"
fi
[ -n "$pcc" ] && [ -x "$pcc" ] || die "nvidia-pcc not found.
  It is x86-64 only and ships in the plain NVIDIA driver .run, e.g.
  NVIDIA-Linux-x86_64-595.99.02, not in the SC package. Set VC_PCC=/path/to/nvidia-pcc."
libdir="${VC_PCC_LIBDIR:-$(dirname "$pcc")}"

# --- stage every JSON under one common prefix -------------------------------
# pcc's -prefix is a filename prefix filter (a run with the vendor's
# "...pipeline_1" compiled only pipeline_1; "...pipeline" compiled both). The
# emitted names are <kernel>.b<X>_<Y>_<Z>.json, which share no prefix across
# kernels, so give them one: the JSON is renamed, the .spv keeps the name the
# JSON refers to.
stage="$(mktemp -d "${TMPDIR:-/tmp}/vksc-cache.XXXXXX")"
trap 'rm -rf "$stage"' EXIT
mkdir -p "$stage/pipeline"
for j in "${jsons[@]}"; do
  cp "$j" "$stage/pipeline/p_$(basename "$j")"
done

# Copy each .spv the JSONs name. (They sit next to the JSONs; the dir was just
# produced by one compiler run.)
for j in "${jsons[@]}"; do
  while read -r spv; do
    [ -f "$srcdir/$spv" ] || die "JSON $j names $spv, which is not in $srcdir"
    cp "$srcdir/$spv" "$stage/pipeline/$spv"
  done < <(python3 - "$j" <<'PY'
import json, sys
doc = json.load(open(sys.argv[1]))
for entry in doc["ComputePipelineState"]["ShaderFileNames"]:
    print(entry["filename"])
PY
)
done

echo "build_cache: ${#jsons[@]} pipeline(s), ${#spvs[@]} binary(ies) -> chip $chip"
for j in "${jsons[@]}"; do
  python3 - "$j" <<'PY'
import json, sys
doc = json.load(open(sys.argv[1]))
st = doc["ComputePipelineState"]
stage = st["ComputePipeline"]["stage"]
pc = st["PipelineLayout"].get("pPushConstantRanges") or []
lay = st["DescriptorSetLayouts"][0]["pSetLayouts_0"]
bind = lay.get("bindingCount", 0)
uuid = bytes(doc["PipelineUUID"]).hex()
name = sys.argv[1].rsplit("/", 1)[-1]
print("  %-24s entry=%-16s bindings=%d pc=%s uuid=%s"
      % (name, stage["pName"], bind,
         ("%d@%d" % (pc[0]["size"], pc[0]["offset"])) if pc else "-", uuid))
PY
done

# --- one pcc run over the whole staged directory ----------------------------
raw="$stage/cache.bin"
echo "build_cache: $pcc -path $stage/pipeline -prefix p_ -chip $chip -out $raw"
LD_LIBRARY_PATH="$libdir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$pcc" -path "$stage/pipeline" -prefix p_ -chip "$chip" -out "$raw"

[ -s "$raw" ] || die "nvidia-pcc produced no cache (see its output above)"

# --- fix the header the device will check -----------------------------------
mkdir -p "$outdir"
out="$outdir/pipeline_cache.bin"
python3 "$here/patch_cache_header.py" --in "$raw" --out "$out" --device-uuid "$uuid"

echo "build_cache: $out"
echo "build_cache: header as the device will read it:"
python3 "$here/patch_cache_header.py" --in "$out" --show | sed 's/^/  /'

cat <<EOF
build_cache: done. The runtime looks for this file at ./pipeline_cache.bin
  (override with VC_SC_PIPELINE_CACHE). Copy it next to the executable:
    tests on the target: VC_SC_PIPELINE_CACHE=/path/pipeline_cache.bin ./app
EOF
