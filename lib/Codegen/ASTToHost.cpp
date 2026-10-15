//===- ASTToHost.cpp - Lower host subset of AST to C++ source -------------===//
//
// Companion to ASTToGLSL: where the GLSL backend lowers the device subset
// (`__global__`/`__device__`) to a compute shader, this backend lowers the
// host subset (`DeviceAttr::Host`/`None`, including `int main()`) to C++.
// CUDA-style launches `kernel<<<grid,block>>>(args)` are translated into
// vcLoadKernel + VCKernelArg[] + vcLaunchKernel sequences. The device SPIR-V
// is embedded as a static uint32_t array so the output is a single .cpp.
//
// Design: host statements/expressions are emitted as near-verbatim C++
// (declarations, control flow, printf, arithmetic pass straight through).
// Only LaunchExpr and the kernel-handle prologue need real translation.
//
//===----------------------------------------------------------------------===//

#include "vc/Codegen/ASTToHost.h"

#include "vc/Frontend/AST.h"
#include "vc/Frontend/Mangle.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdio>
#include <string>

using namespace vc;
using namespace llvm;

namespace {

class HostEmitter {
  raw_ostream &os;
  // Per-kernel SPIR-V modules keyed by device symbol name (the GLSL entry
  // name, e.g. `vadd` or `ns_kernel`). Each is embedded as its own
  // `static const uint32_t __vc_spirv_<name>[]`.
  const std::vector<host::HostSpirvModule> *modules = nullptr;
  // kernelName -> embed variable name (`__vc_spirv_<name>`), built once from
  // `modules` so emitLaunch can pick the right byte array for each launch.
  llvm::StringMap<std::string> spirvVarForKernel;
  // kernelName -> SPIR-V entry-point name passed to vcLoadKernel. Empty
  // `modules` entries resolve to "main" (the GLSL backend's per-kernel .comp
  // convention); the MLIR backend fills this with the kernel symbol name
  // (its single binary carries one OpEntryPoint per kernel so named).
  llvm::StringMap<std::string> entryPointForKernel;
  // Name -> Type* for every variable declared in the host functions being
  // emitted. Used to classify launch args as Pointer vs Scalar: a DeclRefExpr
  // arg whose declaration has a PointerType is a device pointer (vcMalloc
  // handle) and becomes VCKernelArg::Pointer; otherwise it's a by-value
  // scalar becoming VCKernelArg::Scalar.
  StringMap<const Type *> hostVarTypes;
  // Kernel names referenced by any LaunchExpr in main(). Drives the
  // handle-declaration + vcLoadKernel prologue emitted at the top of main.
  StringSet<> launchedKernels;
  // Vulkan SC only: bare `__vc_k_<kernel>` names the host source mentions (e.g.
  // `vcLaunchKernelIndirect(__vc_k_fillIdx, ...)`). Under SC the real handles
  // are per (kernel, block size), so a bare name is only declared as an alias
  // when the source actually uses one — see emitScDescriptors.
  StringSet<> namedHandleRefs;
  // Name -> FunctionDecl for all top-level host functions. Used to complete
  // call sites that omit trailing defaulted parameters (default arguments).
  llvm::StringMap<const FunctionDecl *> hostFuncDecls;
  // Name -> FunctionDecl for every __global__ kernel. A launch arg's
  // classification (Pointer vs Image vs Scalar) must follow the kernel's
  // declared parameter type, not the host variable's type: the host declares a
  // storage-image arg as `VCImage *src` (an opaque RecordType — VCImage is an
  // unknown type name on the host side), so hostVarTypes cannot distinguish it
  // from a pointer. The ImageType lives on the kernel parameter
  // (`texture2D<float> src`). emitLaunch resolves the kernel decl via this map
  // and reads params[idx] to classify each arg by its real device type.
  llvm::StringMap<const FunctionDecl *> kernelDecls;
  // Vulkan SC: the precompiled pipelines, keyed "kernel|blockX|blockY|blockZ" ->
  // the file-scope VCScPipelineDesc variable the launch site loads. Null (the
  // default) means the embed-and-vcLoadKernel path. See emitScDescriptors.
  const std::vector<host::HostScPipeline> *scPipelines = nullptr;
  llvm::StringMap<std::string> scDescVarForSite;

public:
  HostEmitter(raw_ostream &o, const std::vector<host::HostSpirvModule> *mods,
              const std::vector<host::HostScPipeline> *sc = nullptr)
      : os(o), modules(mods), scPipelines(sc) {
    if (scPipelines) {
      for (const auto &p : *scPipelines) {
        std::string key = scSiteKey(p.kernelName, p.blockX, p.blockY, p.blockZ);
        scDescVarForSite[key] = scDescVarName(p.kernelName, p.blockX, p.blockY,
                                              p.blockZ);
      }
    }
    if (modules) {
      for (const auto &m : *modules) {
        // Empty kernel name = legacy single-module path: embed as the generic
        // `__vc_spirv` (no suffix) and register it as the fallback so any
        // launch whose kernel has no dedicated module resolves to it.
        spirvVarForKernel[m.kernelName] =
            m.kernelName.empty() ? "__vc_spirv"
                                 : "__vc_spirv_" + m.kernelName;
        entryPointForKernel[m.kernelName] =
            m.entryPoint.empty() ? "main" : m.entryPoint;
      }
    }
  }

  // Key for one (kernel, block size) launch site. The block size is part of the
  // identity under Vulkan SC: the workgroup size is baked into the precompiled
  // pipeline, so the same kernel launched with a different block size is a
  // different pipeline (and gets its own descriptor + cache entry).
  static std::string scSiteKey(StringRef kernel, uint32_t bx, uint32_t by,
                               uint32_t bz) {
    return (kernel.str() + "|" + std::to_string(bx) + "|" + std::to_string(by) +
            "|" + std::to_string(bz));
  }

  // File-scope name of the descriptor variable for a (kernel, block size) pair.
  static std::string scDescVarName(StringRef kernel, uint32_t bx, uint32_t by,
                                   uint32_t bz) {
    return ("__vc_sc_" + kernel.str() + "_b" + std::to_string(bx) + "_" +
            std::to_string(by) + "_" + std::to_string(bz));
  }

  // File-scope name of the kernel handle for a (kernel, block size) pair. One
  // per pair, not one per kernel: the handle is what the runtime records the
  // loaded pipeline in, and two block sizes are two different precompiled
  // pipelines, so sharing a handle would make the second load look like a repeat
  // of the first (the `if (!handle)` guard would skip it) and dispatch the
  // wrong pipeline.
  static std::string scHandleName(StringRef kernel, uint32_t bx, uint32_t by,
                                  uint32_t bz) {
    return ("__vc_k_" + kernel.str() + "_b" + std::to_string(bx) + "_" +
            std::to_string(by) + "_" + std::to_string(bz));
  }

  bool emit(const TranslationUnit &tu) {
    const FunctionDecl *mainFn = nullptr;
    // First pass: collect host var types across all host functions (including
    // those nested in namespaces) so launch arg classification works even if a
    // var is declared in an outer scope.
    collectHostFuncInfo(tu.decls);

    // Locate the host main() (recursing into namespaces).
    mainFn = findMain(tu.decls);
    if (!mainFn) return false;

    // Find every kernel launch reachable from main so we can preload handles at
    // the top of main. The scan is syntactic, not a call graph: we sweep every
    // host function body, not just main's, because a launch may live in a host
    // helper (`runCase(...)`) that main calls. Over-collecting is harmless — an
    // unused `VCKernelHandle __vc_k_x = nullptr;` costs one pointer, and a
    // launch site whose handle was never declared is a hard compile error.
    if (mainFn->body) scanLaunches(mainFn->body.get());
    for (const auto &kv : hostFuncDecls)
      if (kv.second != mainFn && kv.second->body)
        scanLaunches(kv.second->body.get());

    emitPreamble(tu);
    if (scPipelines) emitScDescriptors();
    else emitSpirvEmbed();
    // Kernel handles live at file scope, not inside main: a launch may sit in a
    // host helper (`runCase(...)`) that main calls, and a helper cannot see a
    // local of main. File scope is also correct for the lazy-load design — the
    // declarations are just null pointers initialized at program start, and the
    // vcLoadKernel guard still runs at the launch site, after the user's
    // vcInit(). See emitKernelHandleDecls.
    emitKernelHandleDecls();

    // Emit top-level struct/typedef/class/namespace declarations so host code
    // can name those types (e.g. `Point hPts[64]`). These are shared with the
    // device subset but the host output needs its own copy visible before main().
    emitHostTypeDecls(tu.decls, /*indent=*/0);

    // Emit host functions (Host/None), preserving namespace blocks. main is one
    // of them; its body gets the kernel-handle prologue prepended.
    emitHostFunctions(tu.decls, /*indent=*/0);
    return true;
  }

  // Recursively collect host function signatures + their local var types, and
  // index them under the qualified name (`ns::f`, `Class::method`) for
  // default-argument completion at call sites.
  void collectHostFuncInfo(const std::vector<NodePtr> &decls) {
    for (auto &d : decls) {
      if (d->getNodeType() == ASTNode::NodeKind::NamespaceDecl) {
        collectHostFuncInfo(static_cast<const NamespaceDecl *>(d.get())->decls);
        continue;
      }
      if (d->getNodeType() != ASTNode::NodeKind::FunctionDecl) continue;
      auto *f = static_cast<const FunctionDecl *>(d.get());
      if (f->deviceAttr == DeviceAttr::Global) {
        // Record kernel decls by bare name so emitLaunch can resolve parameter
        // types for launch-arg classification (see kernelDecls). Scoped kernels
        // (ns::k) are rare and handled at launch via launchHandleName's scope
        // mangling; the bare-name key covers the common case.
        kernelDecls[f->name.str()] = f;
        continue;
      }
      if (f->deviceAttr == DeviceAttr::Device)
        continue;
      hostFuncDecls[hostFuncKey(f)] = f;
      for (ParamDecl *p : f->params)
        if (p->type) hostVarTypes[p->name] = p->type;
      if (f->body) collectHostVars(f->body.get());
    }
  }

  // The host-side spelling of a function name: `ns::f` for a namespace member
  // (recovered from... actually the FunctionDecl doesn't carry its namespace;
  // free functions inside a namespace emit as `ns::f` via the namespace block,
  // so the call site `ns::f(...)` already resolves structurally — we index the
  // bare name too for default-arg completion). A method is `Class::method`.
  static std::string hostFuncKey(const FunctionDecl *f) {
    if (f->isMethod && !f->className.empty())
      return f->className.str() + "::" + f->name.str();
    return f->name.str();
  }

  // Find `int main()` among host functions, recursing into namespaces.
  const FunctionDecl *findMain(const std::vector<NodePtr> &decls) {
    for (auto &d : decls) {
      if (d->getNodeType() == ASTNode::NodeKind::NamespaceDecl)
        if (auto *m = findMain(static_cast<const NamespaceDecl *>(d.get())->decls))
          return m;
      if (d->getNodeType() != ASTNode::NodeKind::FunctionDecl) continue;
      auto *f = static_cast<const FunctionDecl *>(d.get());
      if (f->deviceAttr == DeviceAttr::Global ||
          f->deviceAttr == DeviceAttr::Device)
        continue;
      if (f->name == "main") return f;
    }
    return nullptr;
  }

private:
  // --------------------------------------------------------------------- //
  // Preamble + SPIR-V embedding
  // --------------------------------------------------------------------- //

  void emitPreamble(const TranslationUnit &tu) {
    // User preprocessor lines first (their #includes win for ordering), then
    // the runtime header so VCKernelArg / vc* are always available.
    for (const std::string &line : tu.hostPpLines)
      os << line << "\n";
    os << "#include \"vc/Runtime/VCRuntime.h\"\n";
    os << "#include <cstdio>\n";
    os << "#include <cstring>\n"; // memcpy/memset for host buffer setup
    os << "#include <cmath>\n";   // fabsf/sqrtf/... for host-side verification
    // The runtime symbols (vcInit, VCKernelArg, ...) live in namespace vc.
    // Bring them into scope so the generated launch calls resolve.
    os << "using namespace vc;\n";
    // `dim3` is parsed as a plain identifier (CallExpr); provide a host-side
    // definition so a `dim3(N,N)` expression type-checks even though the launch
    // translator normally splits it into X/Y before emitting it.
    os << "struct dim3 { unsigned x=1,y=1,z=1; "
          "dim3(unsigned X=1,unsigned Y=1,unsigned Z=1):x(X),y(Y),z(Z){} };\n";
    // CUDA-style vector types (float4/int3/half2/...) are device-only in
    // spirit, but a by-value vector can reach host code as a scalar kernel
    // argument (`k<<<1,1>>>(float4(1,2,3,4), out)`). Emit host definitions
    // whose sizeof matches the device's packed layout (elem*count, no padding:
    // float3=12, half2=4, half3=6, half4=8, ...), with x/y/z/w members and a
    // per-count constructor so the ctor call and .x/.y swizzle the host
    // emitter produces verbatim type-check. _Float16 backs `half` (see cppType).
    // make_<vec> aliases map to the same structs.
    os << "struct float2{float x,y;float2(float a=0,float b=0):x(a),y(b){}};"
          "struct float3{float x,y,z;float3(float a=0,float b=0,float c=0)"
          ":x(a),y(b),z(c){}};struct float4{float x,y,z,w;float4(float a=0,"
          "float b=0,float c=0,float d=0):x(a),y(b),z(c),w(d){}};\n";
    os << "struct int2{int x,y;int2(int a=0,int b=0):x(a),y(b){}};"
          "struct int3{int x,y,z;int3(int a=0,int b=0,int c=0):x(a),y(b),"
          "z(c){}};struct int4{int x,y,z,w;int4(int a=0,int b=0,int c=0,"
          "int d=0):x(a),y(b),z(c),w(d){}};\n";
    os << "struct uint2{unsigned x,y;uint2(unsigned a=0,unsigned b=0):x(a),"
          "y(b){}};struct uint3{unsigned x,y,z;uint3(unsigned a=0,unsigned b=0,"
          "unsigned c=0):x(a),y(b),z(c){}};struct uint4{unsigned x,y,z,w;"
          "uint4(unsigned a=0,unsigned b=0,unsigned c=0,unsigned d=0):x(a),"
          "y(b),z(c),w(d){}};\n";
    os << "struct double2{double x,y;double2(double a=0,double b=0):x(a),"
          "y(b){}};struct double3{double x,y,z;double3(double a=0,double b=0,"
          "double c=0):x(a),y(b),z(c){}};struct double4{double x,y,z,w;"
          "double4(double a=0,double b=0,double c=0,double d=0):x(a),y(b),"
          "z(c),w(d){}};\n";
    os << "struct long2{long x,y;long2(long a=0,long b=0):x(a),y(b){}};"
          "struct long3{long x,y,z;long3(long a=0,long b=0,long c=0):x(a),"
          "y(b),z(c){}};struct long4{long x,y,z,w;long4(long a=0,long b=0,"
          "long c=0,long d=0):x(a),y(b),z(c),w(d){}};\n";
    os << "struct ulong2{unsigned long x,y;ulong2(unsigned long a=0,"
          "unsigned long b=0):x(a),y(b){}};struct ulong3{unsigned long x,y,z;"
          "ulong3(unsigned long a=0,unsigned long b=0,unsigned long c=0):x(a),"
          "y(b),z(c){}};struct ulong4{unsigned long x,y,z,w;ulong4(unsigned "
          "long a=0,unsigned long b=0,unsigned long c=0,unsigned long d=0):"
          "x(a),y(b),z(c),w(d){}};\n";
    os << "struct half2{_Float16 x,y;half2(_Float16 a=0,_Float16 b=0):x(a),"
          "y(b){}};struct half3{_Float16 x,y,z;half3(_Float16 a=0,_Float16 b=0,"
          "_Float16 c=0):x(a),y(b),z(c){}};struct half4{_Float16 x,y,z,w;"
          "half4(_Float16 a=0,_Float16 b=0,_Float16 c=0,_Float16 d=0):x(a),"
          "y(b),z(c),w(d){}};\n";
    // Legacy CUDA double-underscore spellings alias the bare names.
    os << "using __half2=half2;using __half3=half3;using __half4=half4;\n";
    // make_<vec>(...) constructor aliases.
    os << "inline float2 make_float2(float a,float b){return float2(a,b);}"
          "inline float3 make_float3(float a,float b,float c){return float3(a,b"
          ",c);}inline float4 make_float4(float a,float b,float c,float d)"
          "{return float4(a,b,c,d);}\n";
    os << "inline int2 make_int2(int a,int b){return int2(a,b);}inline int3 "
          "make_int3(int a,int b,int c){return int3(a,b,c);}inline int4 "
          "make_int4(int a,int b,int c,int d){return int4(a,b,c,d);}\n";
    os << "inline uint2 make_uint2(unsigned a,unsigned b){return uint2(a,b);}"
          "inline uint3 make_uint3(unsigned a,unsigned b,unsigned c){return "
          "uint3(a,b,c);}inline uint4 make_uint4(unsigned a,unsigned b,"
          "unsigned c,unsigned d){return uint4(a,b,c,d);}\n";
    os << "inline double2 make_double2(double a,double b){return double2(a,b);}"
          "inline double3 make_double3(double a,double b,double c){return "
          "double3(a,b,c);}inline double4 make_double4(double a,double b,"
          "double c,double d){return double4(a,b,c,d);}\n";
    os << "inline long2 make_long2(long a,long b){return long2(a,b);}inline "
          "long3 make_long3(long a,long b,long c){return long3(a,b,c);}"
          "inline long4 make_long4(long a,long b,long c,long d){return "
          "long4(a,b,c,d);}\n";
    os << "inline ulong2 make_ulong2(unsigned long a,unsigned long b){return "
          "ulong2(a,b);}inline ulong3 make_ulong3(unsigned long a,unsigned long "
          "b,unsigned long c){return ulong3(a,b,c);}inline ulong4 make_ulong4("
          "unsigned long a,unsigned long b,unsigned long c,unsigned long d)"
          "{return ulong4(a,b,c,d);}\n";
    os << "inline half2 make_half2(_Float16 a,_Float16 b){return half2(a,b);}"
          "inline half3 make_half3(_Float16 a,_Float16 b,_Float16 c){return "
          "half3(a,b,c);}inline half4 make_half4(_Float16 a,_Float16 b,"
          "_Float16 c,_Float16 d){return half4(a,b,c,d);}\n\n";
  }

  // Vulkan SC: instead of embedding SPIR-V, emit one file-scope
  // VCScPipelineDesc per (kernel, block size). The runtime loads the pipeline
  // out of the precompiled pipeline cache by UUID — there is no module in this
  // executable at all.
  //
  // The descriptor's layout fields are filled in by the driver from the same
  // table that produced the pcc JSON, so the layout the runtime hands
  // vkCreateComputePipelines is by construction the layout pcc compiled
  // against; a disagreement there is a bare -1000011001 with no hint of which
  // binding differs.
  void emitScDescriptors() {
    if (!scPipelines || scPipelines->empty()) {
      os << "// no Vulkan SC pipelines (no __global__ kernels)\n\n";
      return;
    }
    os << "// Offline (Vulkan SC) pipelines: one per (kernel, launch block "
          "size).\n"
          "// The SPIR-V these were compiled from is not in this executable; "
          "the\n"
          "// runtime fetches each pipeline from pipeline_cache.bin by "
          "UUID.\n";
    for (const auto &p : *scPipelines) {
      const std::string var =
          scDescVarName(p.kernelName, p.blockX, p.blockY, p.blockZ);
      os << "static const VCScPipelineDesc " << var << " = {\n";
      os << "  \"" << p.kernelName << "\", \"" << p.entryPoint << "\",\n";
      os << "  {";
      for (unsigned i = 0; i < 16; ++i) {
        char buf[8];
        std::snprintf(buf, sizeof(buf), "%s0x%02x", i ? ", " : "",
                      (unsigned)p.uuid[i]);
        os << buf;
      }
      os << "},\n";
      os << "  " << p.blockX << ", " << p.blockY << ", " << p.blockZ << ",\n";
      os << "  " << p.storageBufferBindingCount << ", " << p.imageBindingCount
         << ", " << p.pushConstantSize << ", 0\n";
      os << "};\n";
    }
    os << "\n";
    // Handles live here rather than in emitKernelHandleDecls (which emits one
    // per kernel): under SC the identity is the (kernel, block size) pair.
    os << "// Kernel handles: one per (kernel, block size). A different block "
          "size is a\n"
          "// different precompiled pipeline, so it gets its own handle — the "
          "lazy\n"
          "// guard below would otherwise treat the second load as a repeat of "
          "the first.\n";
    for (const auto &p : *scPipelines)
      os << "static VCKernelHandle "
         << scHandleName(p.kernelName, p.blockX, p.blockY, p.blockZ)
         << " = nullptr;\n";
    // Host code may name the handle the default backends would declare —
    // `vcLaunchKernelIndirect(__vc_k_foo, ...)` or
    // `vcOccupancyMaxPotentialBlockSize(..., __vc_k_foo, ...)`. Under SC a
    // handle is bound to one baked block size, so a kernel's bare name aliases
    // its first (source-order) block size; a named use at another block size is
    // rejected at dispatch with both sizes reported, rather than silently
    // dispatching the wrong pipeline.
    StringSet<> aliased;
    for (const auto &p : *scPipelines) {
      std::string bare = "__vc_k_" + p.kernelName;
      if (!namedHandleRefs.count(bare) || !aliased.insert(bare).second) continue;
      os << "// `" << bare << "` is referenced by name below; under Vulkan SC the\n"
            "// handle carries its block size (" << p.blockX << "x" << p.blockY
         << "x" << p.blockZ
         << ", this kernel's first launch site).\n";
      os << "static VCKernelHandle &" << bare << " = "
         << scHandleName(p.kernelName, p.blockX, p.blockY, p.blockZ) << ";\n";
    }
    os << "\n";
  }

  void emitSpirvEmbed() {
    if (!modules || modules->empty()) {
      os << "// no device SPIR-V (no __global__ kernels)\n\n";
      return;
    }
    os << "// Embedded device SPIR-V (one module per __global__ kernel), "
          "little-endian words.\n";
    for (const auto &m : *modules) {
      const std::string var =
          m.kernelName.empty() ? "__vc_spirv"
                               : "__vc_spirv_" + m.kernelName;
      os << "static const uint32_t " << var << "[] = {";
      for (size_t i = 0; i < m.wordCount; ++i) {
        if ((i % 8) == 0) os << "\n  ";
        else os << " ";
        char buf[16];
        std::snprintf(buf, sizeof(buf), "0x%08x", m.words[i]);
        os << buf;
        if (i + 1 < m.wordCount) os << ",";
      }
      if (m.wordCount) os << "\n";
      os << "};\n";
      os << "static const size_t " << var << "_len = " << m.wordCount << ";\n";
    }
    os << "\n";
  }

  // Emit top-level struct/typedef/enum/constant/class declarations as plain
  // C++, recursing into namespaces (emitting them verbatim). These mirror the
  // device-side definitions; the host needs them visible so host code can name
  // those types (e.g. `Point hPts[64]`).
  void emitHostTypeDecls(const std::vector<NodePtr> &decls, int indent) {
    std::string pad(indent * 2, ' ');
    for (auto &d : decls) {
      if (d->getNodeType() == ASTNode::NodeKind::NamespaceDecl) {
        auto *ns = static_cast<const NamespaceDecl *>(d.get());
        os << pad << "namespace " << ns->name << " {\n";
        emitHostTypeDecls(ns->decls, indent + 1);
        os << pad << "} // namespace " << ns->name << "\n";
      } else if (d->getNodeType() == ASTNode::NodeKind::StructDecl) {
        auto *s = static_cast<const StructDecl *>(d.get());
        if (s->isClass) os << pad << "class " << s->name << " {\n public:\n";
        else os << pad << "struct " << s->name << " {\n";
        for (FieldDecl *f : s->fields) {
          os << pad << "  " << cppType(f->type) << " " << f->name;
          for (int64_t dim : f->arrayDims)
            os << "[" << dim << "]";
          os << ";\n";
        }
        // Class methods: emit in-class declaration only. The method body is
        // lowered as a qualified `Class::method(...)` definition by the host
        // function emit pass (out-of-line). Skip the synthesized `_this` param,
        // which is real C++ `this`.
        for (FunctionDecl *m : s->methods) {
          os << pad << "  " << cppType(m->returnType) << " " << m->name << "(";
          bool first = true;
          for (ParamDecl *p : m->params) {
            if (p->name == "_this") continue;
            if (!first) os << ", ";
            first = false;
            os << cppType(p->type) << " " << p->name;
          }
          os << ");\n";
        }
        os << pad << "};\n";
      } else if (d->getNodeType() == ASTNode::NodeKind::TypedefDecl) {
        auto *td = static_cast<const TypedefDecl *>(d.get());
        // Spell the alias back the way the source wrote it: `using A = T;` vs
        // `typedef T A;` — the host preamble is compiled verbatim by g++.
        if (td->isUsing)
          os << pad << "using " << td->name << " = " << cppType(td->underlying)
             << ";\n";
        else
          os << pad << "typedef " << cppType(td->underlying) << " " << td->name
             << ";\n";
      } else if (d->getNodeType() == ASTNode::NodeKind::EnumDecl) {
        // GLSL has no enum, but C++ does — however emitting the constants as
        // `const int` keeps the host and device backends identical and avoids
        // any scoping surprises (VC supports unscoped enums only). Host code
        // references enum constants by bare name just like device code.
        auto *ed = static_cast<const EnumDecl *>(d.get());
        for (auto &c : ed->constants)
          os << pad << "const int " << c.name << " = " << c.value << ";\n";
      } else if (d->getNodeType() == ASTNode::NodeKind::VarDecl) {
        // __constant__ globals: emit as `const` C++ globals so host code can
        // reference them too. Mirrors the GLSL backend's `const` lowering.
        // File-scope `static`/`extern` globals (storageClass) are also emitted
        // here with the storage class passed through verbatim — these are host
        // translation-unit globals, legal C++.
        // File-scope C `const`/`constexpr` globals (`const int N = 4;`) are
        // emitted as `const` C++ globals too: the host references the same
        // compile-time constants the device does (array sizes, loop bounds).
        // A const global without an initializer is a declaration only
        // (`extern const int g;`) and skipped — there's no value to emit.
        auto *v = static_cast<const VarDecl *>(d.get());
        bool isConstGlobal = v->isConstant || v->isConst;
        bool hasStorage = v->storageClass != StorageClass::None;
        if (!isConstGlobal && !hasStorage) continue;
        if (v->isConst && !v->isConstant && !v->init) continue;
        if (v->storageClass == StorageClass::Static) os << pad << "static ";
        else if (v->storageClass == StorageClass::Extern) os << pad << "extern ";
        if (isConstGlobal) os << "const ";
        os << cppType(v->type) << " " << v->name;
        for (int64_t dim : v->arrayDims)
          os << "[" << dim << "]";
        if (v->init) { os << " = "; emitHostExpr(v->init.get()); }
        os << ";\n";
      }
    }
    if (indent == 0) os << "\n";
  }

  // Emit host functions, preserving namespace blocks. Kernel handles are
  // declared at file scope (see emitKernelHandleDecls), not here.
  void emitHostFunctions(const std::vector<NodePtr> &decls, int indent) {
    std::string pad(indent * 2, ' ');
    for (auto &d : decls) {
      if (d->getNodeType() == ASTNode::NodeKind::NamespaceDecl) {
        auto *ns = static_cast<const NamespaceDecl *>(d.get());
        os << pad << "namespace " << ns->name << " {\n";
        emitHostFunctions(ns->decls, indent + 1);
        os << pad << "} // namespace " << ns->name << "\n";
        continue;
      }
      if (d->getNodeType() != ASTNode::NodeKind::FunctionDecl) continue;
      auto *f = static_cast<const FunctionDecl *>(d.get());
      if (f->deviceAttr == DeviceAttr::Global ||
          f->deviceAttr == DeviceAttr::Device)
        continue;
      os << pad;
      emitHostFunction(f);
    }
  }

  // --------------------------------------------------------------------- //
  // Host var type collection + launch scanning
  // --------------------------------------------------------------------- //

  void collectHostVars(const ASTNode *n) {
    if (!n) return;
    switch (n->getNodeType()) {
    case ASTNode::NodeKind::CompoundStmt:
      for (auto &s : static_cast<const CompoundStmt *>(n)->statements)
        collectHostVars(s.get());
      return;
    case ASTNode::NodeKind::DeclStmt: {
      for (VarDecl *d : static_cast<const DeclStmt *>(n)->decls)
        if (d && d->type) hostVarTypes[d->name] = d->type;
      return;
    }
    case ASTNode::NodeKind::IfStmt: {
      auto *iff = static_cast<const IfStmt *>(n);
      collectHostVars(iff->thenStmt.get());
      collectHostVars(iff->elseStmt.get());
      return;
    }
    case ASTNode::NodeKind::ForStmt: {
      auto *fs = static_cast<const ForStmt *>(n);
      collectHostVars(fs->init.get());
      collectHostVars(fs->body.get());
      return;
    }
    case ASTNode::NodeKind::WhileStmt:
      collectHostVars(static_cast<const WhileStmt *>(n)->body.get());
      return;
    case ASTNode::NodeKind::DoStmt:
      collectHostVars(static_cast<const DoStmt *>(n)->body.get());
      return;
    case ASTNode::NodeKind::SwitchStmt:
      collectHostVars(static_cast<const SwitchStmt *>(n)->body.get());
      return;
    default:
      return;
    }
  }

  void scanLaunches(const ASTNode *n) {
    if (!n) return;
    switch (n->getNodeType()) {
    case ASTNode::NodeKind::DeclRefExpr: {
      // Remember bare kernel-handle references: the default backends name the
      // handle `__vc_k_<kernel>`, and host code that passes one explicitly
      // (`vcLaunchKernelIndirect`, `vcOccupancyMaxPotentialBlockSize`) keeps
      // compiling under SC only if that name exists there too.
      StringRef name = static_cast<const DeclRefExpr *>(n)->name;
      if (name.starts_with("__vc_k_")) namedHandleRefs.insert(name);
      return;
    }
    case ASTNode::NodeKind::LaunchExpr: {
      auto *l = static_cast<const LaunchExpr *>(n);
      if (l->callee)
        launchedKernels.insert(launchHandleName(l->callee.get()));
      scanLaunches(l->gridDim.get());
      scanLaunches(l->blockDim.get());
      scanLaunches(l->gridDimY.get());
      scanLaunches(l->blockDimY.get());
      scanLaunches(l->stream.get());
      for (auto &a : l->args) scanLaunches(a.get());
      return;
    }
    case ASTNode::NodeKind::CompoundStmt:
      for (auto &s : static_cast<const CompoundStmt *>(n)->statements)
        scanLaunches(s.get());
      return;
    case ASTNode::NodeKind::DeclStmt: {
      auto *d = static_cast<const DeclStmt *>(n);
      for (VarDecl *v : d->decls)
        if (v && v->init) scanLaunches(v->init.get());
      return;
    }
    case ASTNode::NodeKind::ExprStmt:
      scanLaunches(static_cast<const ExprStmt *>(n)->expr.get());
      return;
    case ASTNode::NodeKind::ReturnStmt:
      scanLaunches(static_cast<const ReturnStmt *>(n)->value.get());
      return;
    case ASTNode::NodeKind::IfStmt: {
      auto *iff = static_cast<const IfStmt *>(n);
      scanLaunches(iff->cond.get());
      scanLaunches(iff->thenStmt.get());
      scanLaunches(iff->elseStmt.get());
      return;
    }
    case ASTNode::NodeKind::ForStmt: {
      auto *fs = static_cast<const ForStmt *>(n);
      scanLaunches(fs->init.get());
      scanLaunches(fs->cond.get());
      scanLaunches(fs->step.get());
      scanLaunches(fs->body.get());
      return;
    }
    case ASTNode::NodeKind::WhileStmt: {
      auto *ws = static_cast<const WhileStmt *>(n);
      scanLaunches(ws->cond.get());
      scanLaunches(ws->body.get());
      return;
    }
    case ASTNode::NodeKind::DoStmt: {
      auto *ds = static_cast<const DoStmt *>(n);
      scanLaunches(ds->body.get());
      scanLaunches(ds->cond.get());
      return;
    }
    case ASTNode::NodeKind::SwitchStmt: {
      auto *sw = static_cast<const SwitchStmt *>(n);
      scanLaunches(sw->cond.get());
      scanLaunches(sw->body.get());
      return;
    }
    case ASTNode::NodeKind::CaseStmt: {
      auto *cs = static_cast<const CaseStmt *>(n);
      scanLaunches(cs->value.get());
      scanLaunches(cs->sub.get());
      return;
    }
    case ASTNode::NodeKind::BinaryExpr: {
      auto *b = static_cast<const BinaryExpr *>(n);
      scanLaunches(b->lhs.get());
      scanLaunches(b->rhs.get());
      return;
    }
    case ASTNode::NodeKind::UnaryExpr:
      scanLaunches(static_cast<const UnaryExpr *>(n)->operand.get());
      return;
    case ASTNode::NodeKind::ConditionalExpr: {
      auto *c = static_cast<const ConditionalExpr *>(n);
      scanLaunches(c->cond.get());
      scanLaunches(c->thenExpr.get());
      scanLaunches(c->elseExpr.get());
      return;
    }
    case ASTNode::NodeKind::CommaExpr: {
      auto *c = static_cast<const CommaExpr *>(n);
      scanLaunches(c->lhs.get());
      scanLaunches(c->rhs.get());
      return;
    }
    case ASTNode::NodeKind::CStyleCastExpr:
      scanLaunches(static_cast<const CStyleCastExpr *>(n)->sub.get());
      return;
    case ASTNode::NodeKind::InitListExpr:
      for (auto &e : static_cast<const InitListExpr *>(n)->elements)
        scanLaunches(e.get());
      return;
    case ASTNode::NodeKind::CallExpr: {
      auto *c = static_cast<const CallExpr *>(n);
      scanLaunches(c->callee.get());
      for (auto &a : c->args) scanLaunches(a.get());
      return;
    }
    case ASTNode::NodeKind::IndexExpr: {
      auto *ie = static_cast<const IndexExpr *>(n);
      scanLaunches(ie->base.get());
      scanLaunches(ie->index.get());
      return;
    }
    case ASTNode::NodeKind::MemberAccessExpr:
      scanLaunches(static_cast<const MemberAccessExpr *>(n)->base.get());
      return;
    default:
      return;
    }
  }

  // --------------------------------------------------------------------- //
  // Function emission
  // --------------------------------------------------------------------- //

  void emitHostFunction(const FunctionDecl *f) {
    // C storage class (`static`/`extern`) passes through verbatim to C++.
    // `static int helper()` is a legal translation-unit-local host function;
    // `extern` (prototype-only, no body) is also legal C++.
    if (f->storageClass == StorageClass::Static) os << "static ";
    else if (f->storageClass == StorageClass::Extern) os << "extern ";
    os << cppType(f->returnType) << " ";
    // A class method (out-of-line definition `Class::method`) emits its
    // qualified name so g++ sees a real C++ member definition. The body uses
    // `this` natively, which the parser spelled `_this` — rewrite below.
    if (f->isMethod && !f->className.empty())
      os << f->className << "::" << f->name;
    else
      os << f->name;
    os << "(";
    bool first = true;
    for (unsigned i = 0; i < f->params.size(); ++i) {
      // The synthesized `_this` param is the device-side encoding of `this`;
      // a real C++ method has an implicit `this`, so skip it on the host.
      if (f->isMethod && f->params[i]->name == "_this") continue;
      if (!first) os << ", ";
      first = false;
      if (f->params[i]->isConst) os << "const ";
      os << cppType(f->params[i]->type) << " " << f->params[i]->name;
    }
    os << ") {\n";
    if (f->body &&
        f->body->getNodeType() == ASTNode::NodeKind::CompoundStmt) {
      auto *cs = static_cast<const CompoundStmt *>(f->body.get());
      for (auto &s : cs->statements)
        emitHostStmt(s.get(), 1);
    }
    os << "}\n\n";
  }

  // File-scope kernel-handle declarations, emitted once just after the
  // embedded SPIR-V. Every launched kernel gets one null handle; the actual
  // vcLoadKernel call is deferred to a lazy guard at each launch site.
  //
  // Why file scope and not a local at the top of main: vcInit() may be called
  // anywhere in the user's main(), so loading eagerly would run before it and
  // silently fail (loadKernel checks init_). The lazy guard at the launch site
  // necessarily runs after vcInit(), and it needs the handle to be reachable
  // from wherever the launch textually sits — which may be a host helper that
  // main calls, and a helper cannot see a local of main. File scope satisfies
  // both. Zero-initialized statics cost nothing at startup.
  void emitKernelHandleDecls() {
    // SC: one handle per (kernel, block size), emitted with the descriptors
    // (emitScDescriptors) since only there are the block sizes known.
    if (scPipelines) return;
    if (launchedKernels.empty()) return;
    os << "// Kernel handles: one per __global__ launched anywhere in this file.\n";
    for (const auto &kv : launchedKernels)
      os << "static VCKernelHandle __vc_k_" << kv.getKey() << " = nullptr;\n";
    os << "\n";
  }

  void pad(unsigned n) { for (unsigned i = 0; i < n; ++i) os << "  "; }

  // --------------------------------------------------------------------- //
  // Statements
  // --------------------------------------------------------------------- //

  void emitForInit(const ASTNode *n) {
    if (!n) return;
    if (n->getNodeType() == ASTNode::NodeKind::DeclStmt) {
      // for-init must be a single declaration. Emit the base type once, then
      // each declarator with its own pointer `*`s (so `int *a, b` stays
      // correct). Rare in practice but kept correct.
      auto *ds = static_cast<const DeclStmt *>(n);
      bool first = true;
      for (VarDecl *d : ds->decls) {
        if (!d) continue;
        if (first) {
          os << cppBaseType(d->type) << " ";
        } else {
          os << ", ";
        }
        first = false;
        os << std::string(pointerDepth(d->type), '*') << d->name;
        for (int64_t dim : d->arrayDims)
          os << "[" << dim << "]";
        if (d->init) { os << " = "; emitHostExpr(d->init.get()); }
      }
      return;
    }
    if (n->getNodeType() == ASTNode::NodeKind::ExprStmt) {
      if (auto *e = static_cast<const ExprStmt *>(n)->expr.get())
        emitHostExpr(e);
    }
  }

  // Depth of pointer wrapping (0 for non-pointers), for for-init emission.
  static unsigned pointerDepth(const Type *t) {
    unsigned d = 0;
    while (t && t->getKind() == TypeKind::Pointer) {
      ++d;
      t = static_cast<const PointerType *>(t)->pointee;
    }
    return d;
  }

  // The base (non-pointer) type spelling, for shared-type for-init emission.
  std::string cppBaseType(const Type *t) {
    while (t && t->getKind() == TypeKind::Pointer)
      t = static_cast<const PointerType *>(t)->pointee;
    return cppType(t);
  }

  void emitHostStmt(const ASTNode *n, unsigned indent) {
    if (!n) return;
    switch (n->getNodeType()) {
    case ASTNode::NodeKind::CompoundStmt: {
      pad(indent); os << "{\n";
      for (auto &s : static_cast<const CompoundStmt *>(n)->statements)
        emitHostStmt(s.get(), indent + 1);
      pad(indent); os << "}\n";
      break;
    }
    case ASTNode::NodeKind::DeclStmt: {
      auto *ds = static_cast<const DeclStmt *>(n);
      if (ds->decls.empty()) break;
      // Emit each declarator on its own line. (C's comma form `T* a, b;`
      // would make `b` a non-pointer, which differs from the per-declarator
      // pointer the parser attached; separate declarations are always
      // correct and stay in the enclosing scope.)
      for (VarDecl *d : ds->decls) {
        if (!d) continue;
        pad(indent);
        if (d->storageClass == StorageClass::Static) os << "static ";
        else if (d->storageClass == StorageClass::Extern) os << "extern ";
        if (d->isConst) os << "const ";
        os << cppType(d->type) << " " << d->name;
        for (int64_t dim : d->arrayDims)
          os << "[" << dim << "]";
        if (d->init) { os << " = "; emitHostExpr(d->init.get()); }
        os << ";\n";
      }
      break;
    }
    case ASTNode::NodeKind::ExprStmt: {
      auto *es = static_cast<const ExprStmt *>(n);
      // A kernel launch appears wrapped in an ExprStmt (the `;` belongs to the
      // statement). Translate it to the runtime call sequence instead of
      // trying to emit it as a plain expression.
      if (es->expr &&
          es->expr->getNodeType() == ASTNode::NodeKind::LaunchExpr) {
        emitLaunch(static_cast<const LaunchExpr *>(es->expr.get()), indent);
        break;
      }
      pad(indent);
      if (auto *e = es->expr.get())
        emitHostExpr(e);
      os << ";\n";
      break;
    }
    case ASTNode::NodeKind::ReturnStmt:
      pad(indent); os << "return";
      if (auto *r = static_cast<const ReturnStmt *>(n)->value.get()) {
        os << " "; emitHostExpr(r);
      }
      os << ";\n";
      break;
    case ASTNode::NodeKind::IfStmt: {
      auto *iff = static_cast<const IfStmt *>(n);
      pad(indent); os << "if ("; emitHostExpr(iff->cond.get()); os << ") {\n";
      if (iff->thenStmt) emitHostStmt(iff->thenStmt.get(), indent + 1);
      pad(indent); os << "}\n";
      if (iff->elseStmt) {
        pad(indent); os << "else {\n";
        emitHostStmt(iff->elseStmt.get(), indent + 1);
        pad(indent); os << "}\n";
      }
      break;
    }
    case ASTNode::NodeKind::ForStmt: {
      auto *fs = static_cast<const ForStmt *>(n);
      pad(indent); os << "for (";
      if (fs->init) emitForInit(fs->init.get());
      os << "; ";
      if (fs->cond) emitHostExpr(fs->cond.get());
      os << "; ";
      if (fs->step) emitHostExpr(fs->step.get());
      os << ") {\n";
      if (fs->body) emitHostStmt(fs->body.get(), indent + 1);
      pad(indent); os << "}\n";
      break;
    }
    case ASTNode::NodeKind::WhileStmt: {
      auto *ws = static_cast<const WhileStmt *>(n);
      pad(indent); os << "while ("; emitHostExpr(ws->cond.get()); os << ") {\n";
      if (ws->body) emitHostStmt(ws->body.get(), indent + 1);
      pad(indent); os << "}\n";
      break;
    }
    case ASTNode::NodeKind::DoStmt: {
      auto *ds = static_cast<const DoStmt *>(n);
      pad(indent); os << "do {\n";
      if (ds->body) emitHostStmt(ds->body.get(), indent + 1);
      pad(indent); os << "} while ("; emitHostExpr(ds->cond.get()); os << ");\n";
      break;
    }
    case ASTNode::NodeKind::BreakStmt:
      pad(indent); os << "break;\n";
      break;
    case ASTNode::NodeKind::ContinueStmt:
      pad(indent); os << "continue;\n";
      break;
    case ASTNode::NodeKind::SwitchStmt: {
      auto *sw = static_cast<const SwitchStmt *>(n);
      pad(indent); os << "switch (";
      emitHostExpr(sw->cond.get());
      os << ") {\n";
      if (sw->body &&
          sw->body->getNodeType() == ASTNode::NodeKind::CompoundStmt) {
        auto *cs = static_cast<const CompoundStmt *>(sw->body.get());
        for (auto &s : cs->statements) emitHostStmt(s.get(), indent + 1);
      } else if (sw->body) {
        emitHostStmt(sw->body.get(), indent + 1);
      }
      pad(indent); os << "}\n";
      break;
    }
    case ASTNode::NodeKind::CaseStmt: {
      auto *cs = static_cast<const CaseStmt *>(n);
      pad(indent);
      if (cs->value) {
        os << "case "; emitHostExpr(cs->value.get()); os << ":\n";
      } else {
        os << "default:\n";
      }
      if (cs->sub) emitHostStmt(cs->sub.get(), indent + 1);
      break;
    }
    default:
      // Expressions appearing as statements (e.g. LaunchExpr) fall here.
      if (n && n->getNodeType() == ASTNode::NodeKind::LaunchExpr) {
        emitLaunch(static_cast<const LaunchExpr *>(n), indent);
        break;
      }
      break;
    }
  }

  // --------------------------------------------------------------------- //
  // Expressions
  // --------------------------------------------------------------------- //

  void emitHostExpr(const ASTNode *n) {
    if (!n) { os << "/*null*/"; return; }
    switch (n->getNodeType()) {
    case ASTNode::NodeKind::IntegerLiteral: {
      auto *il = static_cast<const IntegerLiteral *>(n);
      os << il->value;
      // Re-attach the `long` suffix so a spilled launch arg deduces the right
      // width, matching a long device parameter.
      if (il->isLong)
        os << "L";
      return;
    }
    case ASTNode::NodeKind::FloatLiteral: {
      auto *fl = static_cast<const FloatLiteral *>(n);
      os << fl->value;
      // Re-attach the single-precision suffix so a spilled launch arg
      // (`auto x = (2.5f)`) deduces `float`, matching a float device
      // parameter. Without it the literal is `double` and the device reads
      // the wrong 4 of 8 bytes.
      if (fl->isFloat32)
        os << "f";
      return;
    }
    case ASTNode::NodeKind::BoolLiteral:
      os << (static_cast<const BoolLiteral *>(n)->value ? "true" : "false");
      return;
    case ASTNode::NodeKind::SizeOfExpr: {
      // C++ has sizeof; emit it verbatim and let g++ compute the size. This is
      // ABI-exact on the host (unlike the GLSL device-side fold).
      const auto *s = static_cast<const SizeOfExpr *>(n);
      os << "sizeof(";
      if (s->isType) {
        os << cppType(s->target);
      } else {
        emitHostExpr(s->sub.get());
      }
      os << ")";
      return;
    }
    case ASTNode::NodeKind::CharLiteral:
      os << static_cast<const CharLiteral *>(n)->value;
      return;
    case ASTNode::NodeKind::StringLiteral:
      emitCString(static_cast<const vc::StringLiteral *>(n)->value);
      return;
    case ASTNode::NodeKind::DeclRefExpr: {
      StringRef name = static_cast<const DeclRefExpr *>(n)->name;
      // The parser lowers `this` to a DeclRefExpr named "_this" (the device
      // backend lowers a method to a free function with a `_this` parameter).
      // On the host a method is a real C++ member, so emit the native `this`.
      if (name == "_this") os << "this";
      else os << name;
      return;
    }
    case ASTNode::NodeKind::BinaryExpr: {
      auto *b = static_cast<const BinaryExpr *>(n);
      os << "(";
      emitHostExpr(b->lhs.get());
      os << " " << binopStr(b->op) << " ";
      emitHostExpr(b->rhs.get());
      os << ")";
      return;
    }
    case ASTNode::NodeKind::UnaryExpr: {
      auto *u = static_cast<const UnaryExpr *>(n);
      const char *op = unaryopStr(u->op);
      // Postfix ++/-- print operand then operator; prefix the reverse.
      bool postfix = (u->op == UnaryOp::PostInc || u->op == UnaryOp::PostDec);
      if (!postfix) os << op;
      emitHostExpr(u->operand.get());
      if (postfix) os << op;
      return;
    }
    case ASTNode::NodeKind::ConditionalExpr: {
      auto *c = static_cast<const ConditionalExpr *>(n);
      os << "(";
      emitHostExpr(c->cond.get());
      os << " ? ";
      emitHostExpr(c->thenExpr.get());
      os << " : ";
      emitHostExpr(c->elseExpr.get());
      os << ")";
      return;
    }
    case ASTNode::NodeKind::CommaExpr: {
      auto *c = static_cast<const CommaExpr *>(n);
      os << "(";
      emitHostExpr(c->lhs.get());
      os << ", ";
      emitHostExpr(c->rhs.get());
      os << ")";
      return;
    }
    case ASTNode::NodeKind::CStyleCastExpr: {
      auto *c = static_cast<const CStyleCastExpr *>(n);
      // Emit functional-cast form: T(expr). Valid for scalars and pointers.
      os << "(" << cppType(c->target) << ")(";
      emitHostExpr(c->sub.get());
      os << ")";
      return;
    }
    case ASTNode::NodeKind::InitListExpr: {
      auto *il = static_cast<const InitListExpr *>(n);
      os << "{";
      for (unsigned i = 0; i < il->elements.size(); ++i) {
        if (i) os << ", ";
        emitHostExpr(il->elements[i].get());
      }
      os << "}";
      return;
    }
    case ASTNode::NodeKind::CallExpr: {
      auto *c = static_cast<const CallExpr *>(n);
      emitHostExpr(c->callee.get());
      os << "(";
      for (unsigned i = 0; i < c->args.size(); ++i) {
        if (i) os << ", ";
        emitHostExpr(c->args[i].get());
      }
      // Complete omitted trailing defaulted parameters from the callee's
      // signature (default arguments), so `f(a)` resolves `void f(int,int=10)`.
      if (c->callee &&
          c->callee->getNodeType() == ASTNode::NodeKind::DeclRefExpr) {
        StringRef callee =
            static_cast<const DeclRefExpr *>(c->callee.get())->name;
        auto fit = hostFuncDecls.find(callee);
        if (fit != hostFuncDecls.end()) {
          const FunctionDecl *calleeFn = fit->second;
          for (unsigned i = c->args.size(); i < calleeFn->params.size(); ++i) {
            if (calleeFn->params[i]->defaultVal) {
              if (i) os << ", ";
              emitHostExpr(calleeFn->params[i]->defaultVal.get());
            }
          }
        }
      }
      os << ")";
      return;
    }
    case ASTNode::NodeKind::IndexExpr: {
      auto *ie = static_cast<const IndexExpr *>(n);
      emitHostExpr(ie->base.get());
      os << "[";
      emitHostExpr(ie->index.get());
      os << "]";
      return;
    }
    case ASTNode::NodeKind::MemberAccessExpr: {
      auto *m = static_cast<const MemberAccessExpr *>(n);
      // On the host a method's `this` is a real pointer, so `_this.field`
      // (parser spelling, value-semantics on device) becomes `this->field`.
      bool baseIsThis =
          m->base && m->base->getNodeType() == ASTNode::NodeKind::DeclRefExpr &&
          static_cast<const DeclRefExpr *>(m->base.get())->name == "_this";
      emitHostExpr(m->base.get());
      // The parser lowers both `.` and `::` to MemberAccessExpr; `isScope`
      // records which. A scoped enum/namespace access (VCMemcpyKind::HostToDevice)
      // must emit `::` — `.` is a hard error for those in C++.
      if (m->isScope) os << "::" << m->member;
      else if (baseIsThis) os << "->" << m->member;
      else os << "." << m->member;
      return;
    }
    default:
      os << "/*unhandled expr " << static_cast<unsigned>(n->getNodeType())
         << "*/";
      return;
    }
  }

  // Emit a C string literal with proper escaping.
  void emitCString(const std::string &s) {
    os << "\"";
    for (char c : s) {
      switch (c) {
      case '"': os << "\\\""; break;
      case '\\': os << "\\\\"; break;
      case '\n': os << "\\n"; break;
      case '\t': os << "\\t"; break;
      case '\r': os << "\\r"; break;
      default:
        if ((unsigned char)c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\x%02x", (unsigned char)c);
          os << buf;
        } else {
          os << c;
        }
        break;
      }
    }
    os << "\"";
  }

  // --------------------------------------------------------------------- //
  // Launch translation: kernel<<<grid,block>>>(args)
  // --------------------------------------------------------------------- //

  // Derive the kernel-handle identifier and recorded key for a launch's
  // callee. A bare `kernel<<<...>>>` uses the kernel name; a scoped
  // `ns::kernel<<<...>>>` (MemberAccessExpr with isScope) is mangled to
  // `ns_kernel` (nested `outer::inner::k` -> `outer_inner_k`) so it forms a
  // valid C++ identifier and matches the key recorded by scanLaunches.
  std::string launchHandleName(const ASTNode *callee) const {
    if (callee &&
        callee->getNodeType() == ASTNode::NodeKind::DeclRefExpr)
      return static_cast<const DeclRefExpr *>(callee)->name.str();
    if (callee &&
        callee->getNodeType() == ASTNode::NodeKind::MemberAccessExpr) {
      // Reuse the device mangling by walking the scope chain.
      std::vector<std::string> parts;
      const MemberAccessExpr *ma =
          static_cast<const MemberAccessExpr *>(callee);
      parts.push_back(ma->member.str());
      const ASTNode *cur = ma->base.get();
      while (cur) {
        if (cur->getNodeType() == ASTNode::NodeKind::MemberAccessExpr) {
          auto *sub = static_cast<const MemberAccessExpr *>(cur);
          parts.push_back(sub->member.str());
          cur = sub->base.get();
        } else if (cur->getNodeType() == ASTNode::NodeKind::DeclRefExpr) {
          parts.push_back(static_cast<const DeclRefExpr *>(cur)->name.str());
          break;
        } else {
          break;
        }
      }
      std::reverse(parts.begin(), parts.end());
      // Reuse the shared scope-name mangler: assemble the chain in "::" form
      // and mangle to underscore form (one rule shared with the device path).
      std::string chain;
      for (auto &p : parts) {
        if (!chain.empty()) chain += "::";
        chain += p;
      }
      return mangleScopeName(chain);
    }
    return "main";
  }

  void emitLaunch(const LaunchExpr *l, unsigned indent) {
    pad(indent);
    os << "{\n";
    // Lazy-load the kernel handle on first launch. vcInit() may be called
    // anywhere in the user's main(); this guard runs after it. The SPIR-V
    // module and entry name are keyed by the kernel's device symbol name
    // (kname), which keys both the embedded `__vc_spirv_<kname>` array and the
    // host-side load handle. The SPIR-V entry point is "main" for the GLSL
    // backend (each kernel is its own .comp module; see ASTToGLSL emitBody)
    // and the kernel symbol name for the MLIR backend (one binary, one
    // OpEntryPoint per kernel so named) — resolved via entryPointForKernel.
    std::string kname = launchHandleName(l->callee.get());
    // Resolve the kernel FunctionDecl so each launch arg is classified by its
    // declared device parameter type, not by the host variable's type. A
    // storage-image param (`texture2D<float>`) is an ImageType on the kernel
    // side but an opaque RecordType (`VCImage*`) on the host side, so the host
    // var type alone cannot tell Pointer from Image.
    const FunctionDecl *kernelFunc = nullptr;
    auto kfit = kernelDecls.find(kname);
    if (kfit != kernelDecls.end()) kernelFunc = kfit->second;
    std::string spirvVar;
    if (spirvVarForKernel.count(kname))
      spirvVar = spirvVarForKernel[kname];
    else if (spirvVarForKernel.count("")) // legacy single-module fallback
      spirvVar = spirvVarForKernel[""];
    else
      spirvVar = "__vc_spirv";
    std::string entryPt =
        entryPointForKernel.count(kname)    ? entryPointForKernel[kname]
        : entryPointForKernel.count("")     ? entryPointForKernel[""]
                                            : "main";
    // The handle this launch site uses: under SC it is per (kernel, block size)
    // — see scHandleName — otherwise the one-per-kernel file-scope handle.
    const std::string handleVar =
        scPipelines ? scHandleName(kname, (uint32_t)l->blockX, (uint32_t)l->blockY,
                                   (uint32_t)l->blockZ)
                    : ("__vc_k_" + kname);
    pad(indent + 1);
    if (scPipelines) {
      // Vulkan SC: no module to embed and no entry point to name here — the
      // descriptor already carries both. Pick the descriptor by block size,
      // because the workgroup size is baked into the precompiled pipeline: a
      // launch with a different block is a different pipeline, and one that was
      // never compiled has no cache entry.
      //
      // vcLaunchKernel's block dims must be the descriptor's, or the runtime
      // rejects the launch before dispatch (getPipeline compares them). Using
      // the folded literals rather than the launch expression guarantees that,
      // and the compiler already refused any launch whose block isn't a
      // compile-time constant (an offline pipeline can't match a runtime value).
      auto sit = scDescVarForSite.find(scSiteKey(kname, (uint32_t)l->blockX,
                                                 (uint32_t)l->blockY,
                                                 (uint32_t)l->blockZ));
      if (sit == scDescVarForSite.end()) {
        // Unreachable when the driver built the table from this translation
        // unit's launch sites; emit a named error rather than a dangling call.
        os << "#error \"no offline pipeline compiled for kernel " << kname
           << " at block " << l->blockX << "x" << l->blockY << "x" << l->blockZ
           << "\"\n";
      } else {
        os << "if (!" << handleVar << ") vcLoadKernelOffline(&" << sit->second
           << ", &" << handleVar << ");\n";
      }
    } else {
      os << "if (!__vc_k_" << kname
         << ") vcLoadKernel(" << spirvVar << ", " << spirvVar
         << "_len, \"" << entryPt << "\", &__vc_k_" << kname << ");\n";
    }
    pad(indent + 1);
    // Spill every scalar (non-pointer) launch arg to a synthetic local before
    // building the __args array. A literal or compound-expression arg is an
    // rvalue, so `&(expr)` (the previous form) fails to compile with "lvalue
    // required as unary '&' operand". Materializing each into a named local
    // gives it an address; `auto` preserves the expression's type (int/float/
    // long/...) so sizeof matches what the device ABI expects. Pointer args
    // (vcMalloc handles) are passed by value and need no spill.
    for (unsigned i = 0; i < l->args.size(); ++i) {
      if (!isPointerArg(l->args[i].get(), kernelFunc, i) &&
          !isImageArg(l->args[i].get(), kernelFunc, i)) {
        pad(indent + 1);
        os << "auto __vc_a" << i << " = (";
        emitHostExpr(l->args[i].get());
        os << ");\n";
      }
    }
    os << "VCKernelArg __args[" << l->args.size() << "] = {";
    for (unsigned i = 0; i < l->args.size(); ++i) {
      if (i) os << ", ";
      emitLaunchArg(l->args[i].get(), i, kernelFunc, indent + 1);
    }
    os << "};\n";

    // Choose the runtime launch API by dimensionality and stream presence:
    //   1D, default stream  -> vcLaunchKernel(k, gridX, blockX, args, n)
    //   1D, explicit stream -> vcLaunchKernelS(k, gridX, blockX, args, n, s)
    //   2D, default stream  -> vcLaunchKernel2D(k, gridX, gridY, blockX, blockY, args, n)
    //   2D, explicit stream -> vcLaunchKernel2DS(k, gridX, gridY, blockX, blockY, args, n, s)
    bool is2D = l->gridDimY || l->blockDimY;
    bool hasStream = static_cast<bool>(l->stream);
    pad(indent + 1);
    os << "vcLaunchKernel";
    if (is2D) os << "2D";
    if (hasStream) os << "S";
    os << "(" << handleVar << ", ";
    emitHostExpr(l->gridDim.get());
    if (is2D) {
      os << ", ";
      if (l->gridDimY) emitHostExpr(l->gridDimY.get());
      else os << "1";
    }
    os << ", ";
    emitHostExpr(l->blockDim.get());
    if (is2D) {
      os << ", ";
      if (l->blockDimY) emitHostExpr(l->blockDimY.get());
      else os << "1";
    }
    os << ", __args, " << l->args.size();
    if (hasStream) {
      os << ", ";
      emitHostExpr(l->stream.get());
    }
    os << ");\n";
    pad(indent);
    os << "}\n";
  }

  // True if `arg` is a pointer-typed DeclRefExpr (a vcMalloc device handle):
  // such args are passed by value, not by address, and need no spill.
  // `kernelFunc`/`idx` let us exclude image params: a `texture2D<T>` kernel
  // param is backed on the host by a `VCImage*` (a PointerType wrapping an
  // opaque RecordType), which would otherwise look exactly like a buffer
  // pointer. If the kernel declares an Image param at `idx`, this is an image,
  // not a buffer pointer.
  bool isPointerArg(const ASTNode *arg, const FunctionDecl *kernelFunc,
                    unsigned idx) {
    if (kernelFunc && idx < kernelFunc->params.size()) {
      const Type *pt = kernelFunc->params[idx]->type;
      if (pt && pt->getKind() == TypeKind::Image) return false;
      // The kernel's declared parameter type is authoritative. Falling back to
      // the host variable's type below only recognises a bare DeclRefExpr, so a
      // device handle reached through an expression — `dIn[0]`, `ptr[i]`, a
      // ternary — used to be misclassified as a Scalar. That spilled the handle
      // into a local and passed its address as push-constant bytes, so the
      // kernel read a garbage pointer at a descriptor slot: an Intel driver
      // crash in pipeline creation. Trust the parameter type instead.
      if (pt && pt->getKind() == TypeKind::Pointer) return true;
    }
    if (!arg || arg->getNodeType() != ASTNode::NodeKind::DeclRefExpr)
      return false;
    auto *d = static_cast<const DeclRefExpr *>(arg);
    auto it = hostVarTypes.find(d->name);
    return it != hostVarTypes.end() && it->second &&
           it->second->getKind() == TypeKind::Pointer;
  }

  // A storage-image launch arg: the host variable is a VCImage* declared from
  // vcCreateImage2D. Like pointer args it is passed by handle value (no spill
  // local), but the runtime binds it as a STORAGE_IMAGE descriptor, not an
  // SSBO, so it needs its own VCKernelArg kind.
  //
  // The host variable is an opaque RecordType (VCImage is unknown on the host),
  // so we cannot tell it from a pointer by the var type alone. The decisive
  // signal is the kernel's declared parameter type at `idx` — if the kernel
  // declares a `texture2D<T>` param there, this arg is an Image regardless of
  // the host spelling. `kernelFunc` is null only when the kernel couldn't be
  // resolved (e.g. a scoped name we didn't index); we then fall back to the
  // host var type, which still works for any arg explicitly typed VCImage*
  // when the frontend happens to record it as ImageType.
  bool isImageArg(const ASTNode *arg, const FunctionDecl *kernelFunc,
                  unsigned idx) {
    if (kernelFunc && idx < kernelFunc->params.size()) {
      const Type *pt = kernelFunc->params[idx]->type;
      if (pt && pt->getKind() == TypeKind::Image) return true;
    }
    if (!arg || arg->getNodeType() != ASTNode::NodeKind::DeclRefExpr)
      return false;
    auto *d = static_cast<const DeclRefExpr *>(arg);
    auto it = hostVarTypes.find(d->name);
    return it != hostVarTypes.end() && it->second &&
           it->second->getKind() == TypeKind::Image;
  }

  // Classify one launch arg and emit its VCKernelArg initializer. `idx` is the
  // arg's position; scalar args were already spilled to `__vc_a<idx>` by
  // emitLaunch, so we take that local's address (always an lvalue).
  // `kernelFunc` (may be null) lets us classify image args by the kernel's
  // declared parameter type rather than the host var's opaque type.
  void emitLaunchArg(const ASTNode *arg, unsigned idx,
                     const FunctionDecl *kernelFunc, unsigned /*indent*/) {
    // Pointer arg: pass the handle value directly (no spill local exists).
    if (isPointerArg(arg, kernelFunc, idx)) {
      os << "{VCKernelArg::Pointer, ";
      emitHostExpr(arg);
      os << ", 0}";
      return;
    }
    // Image arg: pass the VCImage* handle directly (no spill local). The
    // runtime reads the VkImageView from the VCImage and binds a
    // STORAGE_IMAGE descriptor.
    if (isImageArg(arg, kernelFunc, idx)) {
      os << "{VCKernelArg::Image, ";
      emitHostExpr(arg);
      os << ", 0}";
      return;
    }
    // Scalar arg: address + size of the spill local. sizeof(__vc_a<idx>) keeps
    // the expression's real width (int/float/long/...), set by `auto` above.
    os << "{VCKernelArg::Scalar, &__vc_a" << idx << ", sizeof(__vc_a" << idx
       << ")}";
  }

  // --------------------------------------------------------------------- //
  // Type + operator spelling
  // --------------------------------------------------------------------- //

  std::string cppType(const Type *t) {
    if (!t) return "int";
    if (t->getKind() == TypeKind::Builtin) {
      switch (static_cast<const BuiltinType *>(t)->builtin) {
      case BuiltinTypeKind::Void: return "void";
      case BuiltinTypeKind::Bool: return "bool";
      case BuiltinTypeKind::Int32: return "int";
      case BuiltinTypeKind::UInt32: return "unsigned int";
      case BuiltinTypeKind::Int64: return "long";
      case BuiltinTypeKind::UInt64: return "unsigned long";
      case BuiltinTypeKind::Float32: return "float";
      case BuiltinTypeKind::Float64: return "double";
      // Host half type. __half (and `half`) map to the GNU/C11 _Float16, a
      // real 16-bit float, so a host `__half arr[N]` occupies 2*N bytes —
      // matching the device SSBO layout a vcMemcpy(2*N) transfers. (The
      // previous `float` widening made host half arrays 4 bytes/elem, corrupting
      // the byte stream the device read back as packed f16.) _Float16
      // arithmetic/casts work directly in g++ and clang.
      case BuiltinTypeKind::Float16: return "_Float16";
      }
    }
    if (t->getKind() == TypeKind::Pointer) {
      // void* stays void*; otherwise T*. Host needs the real pointer type
      // (unlike GLSL, which strips pointers to SSBO element types).
      const Type *p = static_cast<const PointerType *>(t)->pointee;
      if (p && p->getKind() == TypeKind::Builtin &&
          static_cast<const BuiltinType *>(p)->builtin == BuiltinTypeKind::Void)
        return "void*";
      return cppType(p) + "*";
    }
    if (t->getKind() == TypeKind::Reference) {
      // C++ lvalue reference `T&` (host-only, e.g. helper `const Point &p`).
      return cppType(static_cast<const ReferenceType *>(t)->pointee) + "&";
    }
    // texture2D<T> on the host is a VCImage* handle (created by vcCreateImage2D,
    // destroyed by vcDestroyImage). The element type only matters on the
    // device; the host carries an opaque image handle.
    if (t->getKind() == TypeKind::Image)
      return "VCImage*";
    if (t->getKind() == TypeKind::Vector) {
      // float4 etc. are real host types now (see emitPreamble): a struct with
      // x/y/z/w members and sizeof matching the device's packed layout. Used
      // when a by-value vector is a scalar kernel param (the host spill local
      // `auto __vc_aK = (float4(...))` needs the type to resolve). Build the
      // CUDA spelling from the element kind (UInt64 -> "ulong", not the
      // cppType spelling "unsigned long") + count, mirroring the parser's
      // makeVectorType base table.
      auto *v = static_cast<const VectorType *>(t);
      const char *base = "float";
      if (v->elem->getKind() == TypeKind::Builtin) {
        switch (static_cast<const BuiltinType *>(v->elem)->builtin) {
        case BuiltinTypeKind::Float32: base = "float"; break;
        case BuiltinTypeKind::Int32:   base = "int"; break;
        case BuiltinTypeKind::UInt32:  base = "uint"; break;
        case BuiltinTypeKind::Float64: base = "double"; break;
        case BuiltinTypeKind::Bool:    base = "bool"; break;
        case BuiltinTypeKind::Int64:   base = "long"; break;
        case BuiltinTypeKind::UInt64:  base = "ulong"; break;
        case BuiltinTypeKind::Float16: base = "half"; break;
        }
      }
      return std::string(base) + std::to_string(v->count);
    }
    if (t->getKind() == TypeKind::Record) {
      auto *r = static_cast<const RecordType *>(t);
      return std::string(r->decl->name);
    }
    if (t->getKind() == TypeKind::Typedef) {
      auto *td = static_cast<const TypedefType *>(t);
      return cppType(td->decl->underlying);
    }
    return "int";
  }

  const char *binopStr(BinaryOp op) const {
    switch (op) {
    case BinaryOp::Add: return "+";
    case BinaryOp::Sub: return "-";
    case BinaryOp::Mul: return "*";
    case BinaryOp::Div: return "/";
    case BinaryOp::Mod: return "%";
    case BinaryOp::Assign: return "=";
    case BinaryOp::Lt: return "<";
    case BinaryOp::Gt: return ">";
    case BinaryOp::Le: return "<=";
    case BinaryOp::Ge: return ">=";
    case BinaryOp::Eq: return "==";
    case BinaryOp::NEq: return "!=";
    case BinaryOp::Shl: return "<<";
    case BinaryOp::Shr: return ">>";
    case BinaryOp::And: return "&";
    case BinaryOp::Or: return "|";
    case BinaryOp::Xor: return "^";
    case BinaryOp::LAnd: return "&&";
    case BinaryOp::LOr: return "||";
    }
    return "?";
  }

  const char *unaryopStr(UnaryOp op) const {
    switch (op) {
    case UnaryOp::Neg: return "-";
    case UnaryOp::Not: return "~";
    case UnaryOp::LNot: return "!";
    case UnaryOp::Deref: return "*";
    case UnaryOp::AddrOf: return "&";
    case UnaryOp::PreInc: case UnaryOp::PostInc: return "++";
    case UnaryOp::PreDec: case UnaryOp::PostDec: return "--";
    }
    return "?";
  }
};

} // namespace

namespace vc {
namespace host {

bool translateASTToHost(const TranslationUnit &tu,
                        const std::vector<HostSpirvModule> &modules,
                        llvm::raw_ostream &os,
                        const std::vector<HostScPipeline> *scPipelines) {
  HostEmitter e(os, &modules, scPipelines);
  return e.emit(tu);
}

bool translateASTToHost(const TranslationUnit &tu,
                        const uint32_t *spirvWords, size_t wordCount,
                        llvm::raw_ostream &os) {
  // Legacy single-module overload. Without a kernel name we embed under the
  // generic `__vc_spirv` name and emitLaunch's fallback path loads it with the
  // launch callee's name as the entry point. Real drivers should pass the
  // multi-module overload so each kernel embeds under its own name.
  std::vector<HostSpirvModule> mods;
  if (spirvWords && wordCount)
    mods.push_back({std::string(), spirvWords, wordCount});
  HostEmitter e(os, mods.empty() ? nullptr : &mods);
  return e.emit(tu);
}

} // namespace host
} // namespace vc
