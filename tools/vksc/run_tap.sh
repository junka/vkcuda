#!/bin/bash
scp -q /tmp/vkpipe_tap.so nv118:/tmp/vkpipe_tap.so
ssh nv118 bash -s <<'EOF'
R=/ota/samples/vulkan-sc-samples
C=$R/data/pipeline/computeparticles/pipeline_cache.bin
md5sum $C
cd $R
echo "########## under LD_PRELOAD tap"
LD_PRELOAD=/tmp/vkpipe_tap.so /lib/ld-linux-aarch64.so.1 --library-path /tmp/vksclib \
  ./binaries/Linux_aarch64/vksc_computeparticles > /tmp/vo_tap.txt 2>&1
echo "EXIT=$?"
grep -n 'TAP' /tmp/vo_tap.txt | head -80
echo "---- non-tap lines ----"
grep -vn 'TAP' /tmp/vo_tap.txt | head -30
EOF
