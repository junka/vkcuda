#!/bin/bash
# Ground truth: run the vendor sample with the ORIGINAL vendor cache and with
# our pcc-built combined cache, to see whether the SDK's own pipeline-create
# path accepts pcc output on this device.
set -u
R=/ota/samples/vulkan-sc-samples
C=$R/data/pipeline/computeparticles/pipeline_cache.bin
scp -q /tmp/pccdl/v2/pcc_both_devid0.bin nv118:/tmp/vendor_pcc.bin
ssh nv118 bash -s <<'EOF'
set -x
R=/ota/samples/vulkan-sc-samples
C=$R/data/pipeline/computeparticles/pipeline_cache.bin
cd $R
run() {
  /lib/ld-linux-aarch64.so.1 --library-path /tmp/vksclib \
    ./binaries/Linux_aarch64/vksc_computeparticles > /tmp/vo.txt 2>&1
  echo "EXIT=$?"
  grep -niE 'error|fail|cache|pipeline|match' /tmp/vo.txt | head -25
  tail -5 /tmp/vo.txt
}
echo "########## ORIGINAL vendor cache"
ls -la $C
run
echo "########## pcc combined cache"
cp /tmp/vendor_pcc.bin $C
run
EOF
