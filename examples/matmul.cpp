// examples/matmul.cpp — host demo for the tiled matmul kernel.
//
// Build the kernel first:
//   ./build/tools/vc-glsl/vc-glsl test/matmul.vc -o build/matmul.spv
// Then run:
//   ./build/examples/matmul build/matmul.spv
//
// Computes C = A * B for N x N single-precision matrices, row-major,
// using a 16x16 tiled kernel launched as a 2D grid of 16x16 blocks.

#include "vc/Runtime/VCRuntime.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace vc;

int main(int argc, char **argv) {
  const char *spvPath = argc > 1 ? argv[1] : "matmul.spv";

  const unsigned N = 32;        // matrix dimension (multiple of the tile)
  const unsigned TS = 16;       // tile / block size
  const unsigned blocksX = N / TS;
  const unsigned blocksY = N / TS;

  std::vector<float> hA(N * N), hB(N * N), hC(N * N, 0.0f);
  for (unsigned i = 0; i < N; ++i)
    for (unsigned j = 0; j < N; ++j) {
      hA[i * N + j] = (i == j) ? 1.0f : 0.0f; // identity -> C should equal B
      hB[i * N + j] = 1.0f * (i + j);
    }

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

  void *dA = nullptr, *dB = nullptr, *dC = nullptr;
  size_t bytes = N * N * sizeof(float);
  vcMalloc(&dA, bytes);
  vcMalloc(&dB, bytes);
  vcMalloc(&dC, bytes);
  vcMemcpy(dA, hA.data(), bytes, VCMemcpyKind::HostToDevice);
  vcMemcpy(dB, hB.data(), bytes, VCMemcpyKind::HostToDevice);

  int n = N;
  VCKernelArg args[4] = {
    {VCKernelArg::Pointer, dA, 0},
    {VCKernelArg::Pointer, dB, 0},
    {VCKernelArg::Pointer, dC, 0},
    {VCKernelArg::Scalar, &n, sizeof(int)},
  };

  // 2D launch: grid covers N x N elements with TS x TS blocks.
  if (vcLaunchKernel2D(kernel, N, N, TS, TS, args, 4) != VCError::Success) {
    fprintf(stderr, "launch failed\n");
  }
  vcDeviceSynchronize();
  vcMemcpy(hC.data(), dC, bytes, VCMemcpyKind::DeviceToHost);

  int ok = 1;
  for (unsigned i = 0; i < N && ok; ++i)
    for (unsigned j = 0; j < N; ++j)
      if (hC[i * N + j] != hB[i * N + j]) { ok = 0; break; }
  printf("matmul: %s\n", ok ? "PASS" : "FAIL");
  if (!ok)
    for (unsigned i = 0; i < 4; ++i)
      for (unsigned j = 0; j < 4; ++j)
        printf("  C[%u][%u] = %g (exp %g)\n", i, j, hC[i * N + j],
               hB[i * N + j]);

  vcFree(dA);
  vcFree(dB);
  vcFree(dC);
  vcReleaseKernel(kernel);
  vcShutdown();
  return ok ? 0 : 1;
}
