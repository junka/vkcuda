// examples/block_reduce.cpp — host demo for the blockReduce kernel.
//
// Build the kernel first:
//   ./build/tools/vc-glsl/vc-glsl test/reduce.vc -o build/reduce.spv
// Then run:
//   ./build/examples/block_reduce build/reduce.spv
//
// Sums a 128-element array in 4 blocks of 32 threads each. Each block
// produces one partial sum; we verify them on the host.

#include "vc/Runtime/VCRuntime.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace vc;

int main(int argc, char **argv) {
  const char *spvPath = argc > 1 ? argv[1] : "reduce.spv";

  const unsigned N = 128;
  const unsigned block = 32;
  const unsigned numBlocks = (N + block - 1) / block;

  std::vector<float> hIn(N);
  for (unsigned i = 0; i < N; ++i) hIn[i] = 1.0f * i; // 0,1,2,...,127
  std::vector<float> hOut(numBlocks, 0.0f);

  if (vcInit() != VCError::Success) {
    fprintf(stderr, "vcInit failed\n");
    return 1;
  }
  int devCount = 0;
  vcGetDeviceCount(&devCount);
  printf("devices: %d\n", devCount);

  VCKernelHandle kernel = nullptr;
  if (vcLoadKernelFromFile(spvPath, "main", &kernel) != VCError::Success) {
    fprintf(stderr, "could not load kernel '%s'\n", spvPath);
    vcShutdown();
    return 1;
  }

  void *dIn = nullptr, *dOut = nullptr;
  vcMalloc(&dIn, N * sizeof(float));
  vcMalloc(&dOut, numBlocks * sizeof(float));
  vcMemcpy(dIn, hIn.data(), N * sizeof(float), VCMemcpyKind::HostToDevice);

  int n = N;
  VCKernelArg args[3] = {
    {VCKernelArg::Pointer, dIn, 0},
    {VCKernelArg::Pointer, dOut, 0},
    {VCKernelArg::Scalar, &n, sizeof(int)},
  };

  // gridDim = numBlocks (element count), blockDim = block. The runtime
  // computes workgroup count = ceil(gridDim/blockDim) = numBlocks.
  if (vcLaunchKernel(kernel, N, block, args, 3) != VCError::Success) {
    fprintf(stderr, "launch failed\n");
  }
  vcDeviceSynchronize();
  vcMemcpy(hOut.data(), dOut, numBlocks * sizeof(float),
           VCMemcpyKind::DeviceToHost);

  // Expected partial sums: block b covers [b*block, min((b+1)*block, N)).
  int ok = 1;
  for (unsigned b = 0; b < numBlocks; ++b) {
    unsigned lo = b * block;
    unsigned hi = (b + 1) * block;
    if (hi > N) hi = N;
    float expect = 0.0f;
    for (unsigned i = lo; i < hi; ++i) expect += hIn[i];
    if (hOut[b] != expect) {
      ok = 0;
      printf("  block %u: got %g, exp %g\n", b, hOut[b], expect);
    }
  }
  printf("block_reduce: %s\n", ok ? "PASS" : "FAIL");

  vcFree(dIn);
  vcFree(dOut);
  vcReleaseKernel(kernel);
  vcShutdown();
  return ok ? 0 : 1;
}
