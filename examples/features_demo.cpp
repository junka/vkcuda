// examples/features_demo.cpp — host-side demo for test/features.vc.
//
// Exercises every frontend feature added in the enrichment pass end-to-end:
// modulo, bitwise ops, shifts, compound assignment, ++/--, break/continue,
// do-while, ternary, __device__ helpers, CUDA builtin lowering, hex and
// exponent literals.
//
// Build & run:
//   ./build/tools/vc-glsl/vc-glsl test/features.vc -o build/features.spv
//   ./build/examples/features_demo build/features.spv
//
// The host launches the kernel and compares each output element against a
// reference that mirrors the kernel's logic exactly.

#include "vc/Runtime/VCRuntime.h"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace vc;

// ---- reference implementation (must match features.vc byte-for-logic) ----

static float ref_clampf(float x, float lo, float hi) {
  return x < lo ? lo : (x > hi ? hi : x);
}

// sumodd: sum of odd k with k < upTo, k capped at 31 (break at k>=32).
// Mirrors the do-while loop in the kernel: k starts at 0, is incremented
// to 1 first, then checked.
static int ref_sumodd(int upTo) {
  int s = 0;
  int k = 0;
  do {
    k += 1;
    if (k >= 32) break;
    if ((k & 1) == 0) continue;
    s += k;
  } while (k < upTo);
  return s;
}

static float reference(int i) {
  int mask = 0x1F;            // 31
  int v = i & mask;
  int b = ((i % 4) << 3) | ((i >> 2) & 1);
  b = b ^ 1;
  b = b & ~0;                 // no-op mask
  float lo = 0.0f;
  float hi = 8.0f;
  float s = ref_clampf((float)v, lo, hi) + sinf((float)i);
  float t = i < 4 ? 1.0f : 2.0f;
  int cnt = 0;
  cnt += b;
  cnt++;
  int oddsum = ref_sumodd(i);
  return (float)b + s + t + (float)cnt + (float)oddsum;
}

int main(int argc, char **argv) {
  const char *spvPath = argc > 1 ? argv[1] : "features.spv";

  // One block of N threads so threadIdx.x == i directly.
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

  int n = N;
  VCKernelArg args[2] = {
    {VCKernelArg::Pointer, dOut, 0},
    {VCKernelArg::Scalar, &n, sizeof(int)},
  };

  // Launch exactly one block of N threads: grid=N (element count), block=N
  // -> workgroups = ceil(N/N) = 1, so blockIdx.x == 0 and threadIdx.x == i.
  if (vcLaunchKernel(kernel, N, N, args, 2) != VCError::Success) {
    fprintf(stderr, "launch failed\n");
  }
  vcDeviceSynchronize();

  vcMemcpy(hOut.data(), dOut, bytes, VCMemcpyKind::DeviceToHost);

  int ok = 1;
  for (unsigned i = 0; i < N; ++i) {
    float expect = reference(i);
    // Allow a small epsilon for float sin() differences across drivers.
    if (std::fabs(hOut[i] - expect) > 1e-3f) {
      ok = 0;
      if (i < 8)
        printf("  [%u] got %g, exp %g\n", i, hOut[i], expect);
    }
  }
  printf("features: %s\n", ok ? "PASS" : "FAIL");

  vcFree(dOut);
  vcReleaseKernel(kernel);
  vcShutdown();
  return ok ? 0 : 1;
}
