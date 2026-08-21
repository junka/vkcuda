// examples/struct_demo.cpp — host-side demo for test/struct.vc.
//
// Exercises struct definition, typedef alias, struct field access, array
// fields, struct passed as a kernel SSBO, and a __device__ helper taking a
// typedef'd type.
//
// Build & run:
//   ./build/tools/vc-glsl/vc-glsl test/struct.vc -o build/struct.spv
//   ./build/examples/struct_demo build/struct.spv
//
// The kernel: out[i] = w*pts[i].x + w*pts[i].y + (pts[i].v[0] + pts[i].v[1])
// We mirror that here and compare.

#include "vc/Runtime/VCRuntime.h"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace vc;

// Must match the GLSL struct Point { float x; float y; float v[2]; } under
// std430 SSBO layout: scalars/arrays are tightly packed with no padding, so
// the struct is 16 bytes (x=0, y=4, v[0]=8, v[1]=12).
struct Point {
  float x;
  float y;
  float v[2];
};

static float reference(const Point &p, float w) {
  return w * p.x + w * p.y + (p.v[0] + p.v[1]);
}

int main(int argc, char **argv) {
  const char *spvPath = argc > 1 ? argv[1] : "struct.spv";

  const unsigned N = 64;
  std::vector<Point> hPts(N);
  std::vector<float> hOut(N, 0.0f);
  for (unsigned i = 0; i < N; ++i) {
    hPts[i].x = (float)i;
    hPts[i].y = (float)(i * 2);
    hPts[i].v[0] = 1.0f;
    hPts[i].v[1] = 2.0f;
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

  void *dPts = nullptr, *dOut = nullptr;
  size_t ptsBytes = N * sizeof(Point);
  size_t outBytes = N * sizeof(float);
  vcMalloc(&dPts, ptsBytes);
  vcMalloc(&dOut, outBytes);
  vcMemcpy(dPts, hPts.data(), ptsBytes, VCMemcpyKind::HostToDevice);

  float w = 0.5f;
  VCKernelArg args[3] = {
    {VCKernelArg::Pointer, dPts, 0},
    {VCKernelArg::Pointer, dOut, 0},
    {VCKernelArg::Scalar, &w, sizeof(float)},
  };

  // One block of N threads so threadIdx.x == i directly.
  if (vcLaunchKernel(kernel, N, N, args, 3) != VCError::Success) {
    fprintf(stderr, "launch failed\n");
  }
  vcDeviceSynchronize();
  vcMemcpy(hOut.data(), dOut, outBytes, VCMemcpyKind::DeviceToHost);

  int ok = 1;
  for (unsigned i = 0; i < N; ++i) {
    float expect = reference(hPts[i], w);
    if (std::fabs(hOut[i] - expect) > 1e-4f) {
      ok = 0;
      if (i < 8)
        printf("  [%u] got %g, exp %g\n", i, hOut[i], expect);
    }
  }
  printf("struct: %s\n", ok ? "PASS" : "FAIL");

  vcFree(dPts);
  vcFree(dOut);
  vcReleaseKernel(kernel);
  vcShutdown();
  return ok ? 0 : 1;
}
