// examples/vector_add.cpp — host-side demo for the VC language.
//
// Expected flow:
//   1. ./build/bin/vc test/vadd.vc -emit=spirv -o build/vadd.spv
//   2. ./build/bin/vector_add build/vadd.spv
//
// The host allocates two input buffers and one output buffer via the
// CUDA-style runtime API, launches the kernel, copies the result back,
// and prints it.
//
// NOTE: until the VC->GPU lowering pass is implemented, the SPIR-V produced
// by `vc` may be incomplete. The runtime itself (buffer mgmt, pipeline
// construction, dispatch) is fully functional given a valid .spv.

#include "vc/Runtime/VCRuntime.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace vc;

int main(int argc, char **argv) {
  const char *spvPath = argc > 1 ? argv[1] : "vadd.spv";

  const unsigned N = 64;
  std::vector<float> hA(N), hB(N), hC(N, 0.0f);
  for (unsigned i = 0; i < N; ++i) { hA[i] = 1.0f * i; hB[i] = 2.0f; }

  if (vcInit() != VCError::Success) {
    fprintf(stderr, "vcInit failed\n");
    return 1;
  }
  int devCount = 0;
  vcGetDeviceCount(&devCount);
  printf("devices: %d\n", devCount);

  VCKernelHandle kernel = nullptr;
  if (vcLoadKernelFromFile(spvPath, "main", &kernel) != VCError::Success) {
    fprintf(stderr,
            "could not load kernel '%s'.\n"
            "  Did you run: ./build/bin/vc test/vadd.vc -emit=spirv -o vadd.spv\n"
            "  (VC->GPU lowering is still scaffolded; SPIR-V may be incomplete.)\n",
            spvPath);
    vcShutdown();
    return 1;
  }

  void *dA = nullptr, *dB = nullptr, *dC = nullptr;
  size_t bytes = N * sizeof(float);
  vcMalloc(&dA, bytes);
  vcMalloc(&dB, bytes);
  vcMalloc(&dC, bytes);

  vcMemcpy(dA, hA.data(), bytes, VCMemcpyKind::HostToDevice);
  vcMemcpy(dB, hB.data(), bytes, VCMemcpyKind::HostToDevice);

  // Pass `n` as a scalar by value through a small uniform binding.
  int n = N;
  VCKernelArg args[4] = {
    {VCKernelArg::Pointer, dA, 0},
    {VCKernelArg::Pointer, dB, 0},
    {VCKernelArg::Pointer, dC, 0},
    {VCKernelArg::Scalar, &n, sizeof(int)},
  };

  unsigned block = 32;
  if (vcLaunchKernel(kernel, N, block, args, 4) != VCError::Success) {
    fprintf(stderr, "launch failed\n");
  }
  vcDeviceSynchronize();

  vcMemcpy(hC.data(), dC, bytes, VCMemcpyKind::DeviceToHost);

  int ok = 1;
  for (unsigned i = 0; i < N; ++i) {
    float expect = hA[i] + hB[i];
    if (hC[i] != expect) { ok = 0; }
  }
  printf("vector_add: %s\n", ok ? "PASS" : "FAIL");
  if (!ok) for (unsigned i = 0; i < 8; ++i)
    printf("  [%u] %g + %g = %g (exp %g)\n", i, hA[i], hB[i], hC[i],
           hA[i] + hB[i]);

  vcFree(dA);
  vcFree(dB);
  vcFree(dC);
  vcReleaseKernel(kernel);
  vcShutdown();
  return ok ? 0 : 1;
}
