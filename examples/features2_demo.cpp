// examples/features2_demo.cpp — host-side demo for test/features2.vc.
//
// Exercises the second enrichment pass end-to-end in one kernel:
//   - vector type float4 + swizzle (.x .y .z .w, .xz.x .xz.y)
//   - C-style cast (float)i and functional cast float(...)
//   - struct Vec4 + array field, typedef scalar_t, __device__ helper
//   - char literal 'A'
//   - real pre/post ++ -- value semantics (j = a++ yields OLD value)
//   - switch / case / default
//
// Build & run:
//   ./build/tools/vc-glsl/vc-glsl test/features2.vc -o build/features2.spv
//   ./build/examples/features2_demo build/features2.spv

#include "vc/Runtime/VCRuntime.h"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace vc;

// Mirrors features2.vc logic exactly. total = 217 + 2*i + s, where
// s = 100/200/300 for i%3 == 0/1/2.
static float reference(int i) {
  float sumcomp = 1.0f + 2.0f + 3.0f + 4.0f;        // 10
  float sw = 10.0f + 20.0f + 30.0f + 40.0f;         // 100
  float swz = 10.0f + 30.0f;                        // 40 (q.x + q.z)
  float cf = (float)i + (float)i;                   // 2*i
  float ch = (float)'A';                            // 65
  int a = 0;
  int j = a++;                                       // 0 (old value)
  int k = ++a;                                       // 2 (new value)
  int mode = i % 3;
  int s = (mode == 0) ? 100 : (mode == 1) ? 200 : 300;
  return sumcomp + sw + swz + cf + ch + (float)j + (float)k + (float)s;
}

int main(int argc, char **argv) {
  const char *spvPath = argc > 1 ? argv[1] : "features2.spv";

  const unsigned N = 64;
  std::vector<float> hOut(N, 0.0f);

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

  void *dOut = nullptr;
  size_t bytes = N * sizeof(float);
  vcMalloc(&dOut, bytes);

  // The kernel has no scalar params (output pointer only).
  VCKernelArg args[1] = {
    {VCKernelArg::Pointer, dOut, 0},
  };

  // One block of N threads so threadIdx.x == i directly.
  if (vcLaunchKernel(kernel, N, N, args, 1) != VCError::Success) {
    fprintf(stderr, "launch failed\n");
  }
  vcDeviceSynchronize();
  vcMemcpy(hOut.data(), dOut, bytes, VCMemcpyKind::DeviceToHost);

  int ok = 1;
  for (unsigned i = 0; i < N; ++i) {
    float expect = reference((int)i);
    if (std::fabs(hOut[i] - expect) > 1e-3f) {
      ok = 0;
      if (i < 8)
        printf("  [%u] got %g, exp %g\n", i, hOut[i], expect);
    }
  }
  printf("features2: %s\n", ok ? "PASS" : "FAIL");

  vcFree(dOut);
  vcReleaseKernel(kernel);
  vcShutdown();
  return ok ? 0 : 1;
}
