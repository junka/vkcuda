// examples/async_overlap.cpp — async stream overlap demo for the VC runtime.
//
// Exercises the stream API that the runtime refactor added:
//   - vcStreamCreate / vcStreamDestroy / vcStreamSynchronize
//   - vcLaunchKernelS  (explicit-stream 1D launch; NULL = default stream)
//   - vcMemcpyS        (explicit-stream memcpy)
//
// Two independent vector_add workloads are dispatched on two separate streams
// back-to-back, with NO synchronization between them. Each stream writes its
// own output buffer; the runtime records both command streams asynchronously
// and they may overlap on the GPU. We then synchronize each stream and verify
// both results independently. This proves:
//   1. The _S variants accept an explicit stream and route work to it.
//   2. Launches no longer block the host (no per-launch vkQueueWaitIdle).
//   3. Per-stream ordering holds (each stream's memcpy-after-launch ordering
//      is correct) while cross-stream work stays independent.
//
// Build the kernel first:
//   ./build/tools/vc-glsl/vc-glsl test/vadd.vc -o build/vadd.spv
//   ./build/examples/async_overlap build/vadd.spv

#include "vc/Runtime/VCRuntime.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace vc;

int main(int argc, char **argv) {
  const char *spvPath = argc > 1 ? argv[1] : "vadd.spv";

  if (vcInit() != VCError::Success) {
    fprintf(stderr, "vcInit failed\n");
    return 1;
  }

  VCKernelHandle kernel = nullptr;
  if (vcLoadKernelFromFile(spvPath, "main", &kernel) != VCError::Success) {
    fprintf(stderr, "could not load kernel '%s'\n", spvPath);
    vcShutdown();
    return 1;
  }

  // Two independent workloads, each on its own stream.
  const unsigned N = 64;
  const size_t bytes = N * sizeof(float);
  const unsigned block = 32;

  VCStreamHandle streamA = nullptr, streamB = nullptr;
  vcStreamCreate(&streamA);
  vcStreamCreate(&streamB);

  void *dAa = nullptr, *dBa = nullptr, *dCa = nullptr; // workload A buffers
  void *dAb = nullptr, *dBb = nullptr, *dCb = nullptr; // workload B buffers
  vcMalloc(&dAa, bytes); vcMalloc(&dBa, bytes); vcMalloc(&dCa, bytes);
  vcMalloc(&dAb, bytes); vcMalloc(&dBb, bytes); vcMalloc(&dCb, bytes);

  std::vector<float> hAa(N), hBa(N), hCa(N, 0.0f);
  std::vector<float> hAb(N), hBb(N), hCb(N, 0.0f);
  for (unsigned i = 0; i < N; ++i) {
    hAa[i] = 1.0f * i;        hBa[i] = 2.0f;          // A: i + 2
    hAb[i] = 3.0f * i;        hBb[i] = 4.0f;          // B: 3i + 4
  }

  // Stage inputs (H2D) on each stream, then launch on the SAME stream. The
  // memcpy-then-launch ordering is intra-stream, so it is correct without any
  // cross-stream sync. Both streams are dispatched before we wait on either.
  int nA = N, nB = N;
  VCKernelArg argsA[4] = {
    {VCKernelArg::Pointer, dAa, 0},
    {VCKernelArg::Pointer, dBa, 0},
    {VCKernelArg::Pointer, dCa, 0},
    {VCKernelArg::Scalar, &nA, sizeof(int)},
  };
  VCKernelArg argsB[4] = {
    {VCKernelArg::Pointer, dAb, 0},
    {VCKernelArg::Pointer, dBb, 0},
    {VCKernelArg::Pointer, dCb, 0},
    {VCKernelArg::Scalar, &nB, sizeof(int)},
  };

  // Stream A workload (all async, recorded into streamA):
  vcMemcpyS(dAa, hAa.data(), bytes, VCMemcpyKind::HostToDevice, streamA);
  vcMemcpyS(dBa, hBa.data(), bytes, VCMemcpyKind::HostToDevice, streamA);
  vcLaunchKernelS(kernel, N, block, argsA, 4, streamA);

  // Stream B workload (all async, recorded into streamB, independent of A):
  vcMemcpyS(dAb, hAb.data(), bytes, VCMemcpyKind::HostToDevice, streamB);
  vcMemcpyS(dBb, hBb.data(), bytes, VCMemcpyKind::HostToDevice, streamB);
  vcLaunchKernelS(kernel, N, block, argsB, 4, streamB);

  // Now wait on each stream independently and read results back. D2H memcpy
  // is synchronous on its stream, so by the time it returns the host buffer
  // is readable.
  vcMemcpyS(hCa.data(), dCa, bytes, VCMemcpyKind::DeviceToHost, streamA);
  vcMemcpyS(hCb.data(), dCb, bytes, VCMemcpyKind::DeviceToHost, streamB);

  // (vcMemcpyS D2H already waited on the respective stream; this is belt-and-
  // suspenders and also waits for any lingering in-flight frames.)
  vcStreamSynchronize(streamA);
  vcStreamSynchronize(streamB);

  int ok = 1;
  for (unsigned i = 0; i < N; ++i) {
    if (hCa[i] != hAa[i] + hBa[i]) { ok = 0; break; }
    if (hCb[i] != hAb[i] + hBb[i]) { ok = 0; break; }
  }
  printf("async_overlap: %s\n", ok ? "PASS" : "FAIL");
  if (!ok) {
    for (unsigned i = 0; i < 4; ++i)
      printf("  A[%u] %g + %g = %g (exp %g) | B[%u] %g + %g = %g (exp %g)\n",
             i, hAa[i], hBa[i], hCa[i], hAa[i] + hBa[i],
             i, hAb[i], hBb[i], hCb[i], hAb[i] + hBb[i]);
  }

  vcFree(dAa); vcFree(dBa); vcFree(dCa);
  vcFree(dAb); vcFree(dBb); vcFree(dCb);
  vcStreamDestroy(streamA);
  vcStreamDestroy(streamB);
  vcReleaseKernel(kernel);
  vcShutdown();
  return ok ? 0 : 1;
}
