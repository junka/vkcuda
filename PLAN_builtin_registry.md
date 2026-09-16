# Plan: 抽出统一 BuiltinRegistry

## 目标
消除"同一个 builtin 名字在 Sema / GLSL / MLIR 三处各维护一份列表"的重复，让
**加一个 builtin 只改一处**。这是 P3A builtin overload 的前置：之后 builtin 的签名/
重载集合（`sin` vs `sinf` vs `__sinf`）将集中在一处声明。

## 现状（已直接读取确认）

三处 builtin 信息**形态不同**——这不是 bug，是各站点关注点不同：

| 站点 | 维护什么 | 用途 | 是否可共享 |
|------|---------|------|-----------|
| Sema `isMathBuiltin` (Sema.cpp:89) | ~97 个名字数组 + `make_`/vector 模式 | call 解析时判"是否 builtin"→跳过 undeclared 警告 | **名字集合可共享** |
| Sema `isThreadBuiltin` (Sema.cpp:82) | 5 个名字（threadIdx/blockIdx/blockDim/gridDim/warpSize） | DeclRef 解析时判"是否隐式 builtin" | **名字集合可共享** |
| GLSL `lowerBuiltinCall`/`isAtomicName`/`isWarpIntrinsicName`/`isAsyncCopyBuiltin`/`lowerAtomicName` | 名字分类 + lowering 重写 | 产出 GLSL 字符串 | lowering 逻辑**不可共享**（产出字符串） |
| MLIR `emitMathBuiltin`/`emitAtomicBuiltin`/`emitFenceBuiltin`/`emitVoteBuiltin`/`emitWarpBuiltin`/`emitAsyncCopyBuiltin` | 名字 → spirv op 生成 | 产出 spirv::Op | lowering 逻辑**不可共享**（产出 IR） |

### 关键发现
1. **后端没有独立的"是否 builtin"判断**。GLSL/MLIR 只做 lowering 分发（按顺序尝试，
   不匹配就 fallback 到 user function / math pass-through）。只有 Sema 需要"是否
   builtin"的 bool（两处：call 解析 `isMathBuiltin`，DeclRef 解析 `isThreadBuiltin`）。
2. **lowering 逻辑不可共享**：GLSL 产出字符串，MLIR 产出 spirv::Op，形态完全不同。
   强行抽到 registry 会变成一个胖 switch，反而更难维护。
3. **真实可共享的痛点是"名字集合"**：Sema 的 `isMathBuiltin` 列表必须和后端识别的
   builtin 集合保持同步——Sema 漏认→假 "undeclared function" 警告；后端漏认→lowering
   失败。这正是"加一个 builtin 改三处"的根因。
4. **列表长度不同是合理的**：Sema 全包（math+atomics+sync+warp+async+dim3+vector），
   MLIR math 只列 45（atomics/warp/sync 在各自 emit 函数），GLSL lowerBuiltinCall 只列
   需改名的 20（大部分 math 名字和 GLSL 一样直接 pass-through）。**不强行统一长度**。

## 设计：Registry 只做"名字注册 + 分类"，不做 lowering

```cpp
// include/vc/Frontend/BuiltinRegistry.h
namespace vc {

enum class BuiltinClass {
  Math,        // sin/cos/sqrt/__sinf/sinf/... + geometric (dot/cross/...)
  ThreadIndex, // threadIdx/blockIdx/blockDim/gridDim/warpSize
  Atomic,      // atomicAdd/atomicSub/.../atomicCAS
  Sync,        // __syncthreads/__threadfence*/__syncthreads_count/and/or
  Warp,        // __syncwarp/__ballot_sync/__shfl_*/__anySync/__allSync/__activemask
  AsyncCopy,   // vcMemcpyAsync/vcPipeline*
  VectorCtor,  // float4/int3/.../make_float4/...
  Dim,         // dim3
  Other,       // printf
};

/// True if `name` is any recognized builtin (Sema uses this to skip the
/// undeclared-function warning). Covers all classes above.
bool isBuiltin(llvm::StringRef name);

/// The class of a builtin name, or None if not a builtin. Backends use this
/// to dispatch lowering (each class has its own emit path in GLSL/MLIR).
llvm::Optional<BuiltinClass> builtinClass(llvm::StringRef name);

} // namespace vc
```

### 为什么是这个形状
- `isBuiltin(name)` 取代 Sema 的 `isMathBuiltin` + `isThreadBuiltin`（合并成一个"是
  否任何 builtin"的判断，因为 Sema 两处都只需要 bool）。
- `builtinClass(name)` 让后端**保留各自的 lowering 分发**，但用统一的分类入口替代
  散落的的名字 `if` 链。例如 GLSL 的 `isAtomicName(name)` → `builtinClass(name) ==
  Atomic`；MLIR 的 `emitAtomicBuiltin` 入口判断同理。
- **不把 lowering 逻辑搬进 registry**。GLSL 仍用 `lowerBuiltinCall` 产字符串，MLIR 仍
  用 `emitXxxBuiltin` 产 IR。Registry 只负责"名字→分类"，lowering 留在后端。
- 签名/重载集合**本步不加**——那是 P3A builtin overload 的下一步。现在只集中名字。

## 实施步骤

### Step 1: 新增 `include/vc/Frontend/BuiltinRegistry.h` + `lib/Frontend/BuiltinRegistry.cpp`
- 实现 `isBuiltin` / `builtinClass`。
- 名字列表从 Sema `isMathBuiltin` 的数组搬过来（那是全包版本），按 BuiltinClass 分组。
- vector ctor / make_ 模式匹配搬过来（复用 Sema 的 `isVectorCtorName` 逻辑）。
- 加进 VCFrontend CMake。

### Step 2: Sema 改用 Registry
- 删 `isMathBuiltin`（Sema.cpp:89）和 `isThreadBuiltin`（Sema.cpp:82）的本地名字列表，
  改调 `isBuiltin(name)`。
- 保留 `isVectorCtorName`（它还被别处用？查证；若只被 isMathBuiltin 用则搬进 Registry）。
- call 解析（Sema.cpp:1054）`isMathBuiltin(calleeName)` → `isBuiltin(calleeName)`。
- DeclRef 解析（Sema.cpp:857）`isThreadBuiltin(d->name)` → `builtinClass(d->name) ==
  BuiltinClass::ThreadIndex`（保留 warpSize 的 Int32 类型特判）。

### Step 3: GLSL 改用 Registry 的分类入口
- `isWarpIntrinsicName` → `builtinClass == Warp`
- `isAtomicName` → `builtinClass == Atomic`
- `isAsyncCopyBuiltin` → `builtinClass == AsyncCopy`
- `lowerBuiltinCall` / `lowerAtomicName` / `emitWarpIntrinsic` / `emitAtomicCall` /
  `emitVoteCall` / `emitAsyncCopyCall` 的**lowering 逻辑保留不动**，只把入口的"是否
  X 类"判断换成 Registry。
- 这一步是纯重写入口判断，不改 lowering 行为。

### Step 4: MLIR 改用 Registry 的分类入口
- `emitBuiltinCall`（ASTToMLIR.cpp:2959）的子分发保留，但各 `emitXxxBuiltin` 入口的
  "是否 X 类"判断换成 Registry（如果它们当前是内联名字判断）。
- 同样只改入口判断，不改 lowering。

### Step 5: 验证
- 全量 build。
- lit suite（32 个）全过。
- e2e（66 对）全过（除预存在 printf glsl 失败）。
- 关键回归：atomics.vc、builtins.vc、geometric.vc、math_builtins3.vc、warp 相关 demo
  在两个后端都 PASS——这些直接覆盖各类 builtin 的 lowering。

## 风险
1. **vector ctor / make_ 模式匹配的归属**：Sema `isMathBuiltin` 调 `isVectorCtorName`，
   后端（GLSL `glslVectorCtorName`）有自己的 vector ctor 识别。Registry 的
   `isBuiltin` 需要包含 vector ctor 模式，但后端的 vector ctor **lowering** 留在原处。
   确认 `isVectorCtorName` 是否被 Sema 其他地方用——若只被 isMathBuiltin 用，整个搬进
   Registry；若被别处用，Registry 复制一份逻辑或暴露公共 helper。
2. **`builtinClass` 返回 Optional 的依赖**：VC 用 LLVM，用 `llvm::Optional`（MLIR 18 时代）
   或 `std::optional`（C++20）。确认代码库惯例。
3. **行为零变化**：本步是纯去重，不改任何 builtin 的识别范围或 lowering。若某个名字
   在 Sema 列表但不在 Registry（遗漏），会变成假 undeclared 警告——必须保证 Registry
   的名字集合 ⊇ 原 Sema 列表。

## 不做
- 不加 builtin 签名/重载集合（P3A 下一步）。
- 不把 lowering 逻辑搬进 Registry（GLSL 产字符串、MLIR 产 IR，留在后端）。
- 不改任何 builtin 的 lowering 行为。
- 不动 `isVectorCtorName` 的 vector ctor 类型识别逻辑（只搬位置）。

## 交付物
- 新文件：`include/vc/Frontend/BuiltinRegistry.h`、`lib/Frontend/BuiltinRegistry.cpp`
- 改动：Sema.cpp（删本地列表，改调 Registry）、ASTToGLSL.cpp（入口判断改 Registry）、
  ASTToMLIR.cpp（入口判断改 Registry）
- CMake：`lib/Frontend/CMakeLists.txt` 加 BuiltinRegistry.cpp
- 验证：lit 32/32 + e2e 66/66（除预存在失败）

## Status: DONE

All steps executed. Verification results:

- New files: `include/vc/Frontend/BuiltinRegistry.h`, `lib/Frontend/BuiltinRegistry.cpp`
  (added to VCFrontend in CMake).
- Registry exposes `isBuiltin(name)`, `builtinClass(name)` (returns BuiltinClass
  enum with a `None` value — no Optional dependency), and `isVectorCtorName(name)`.
- Replaced duplicate classification sites:
  - Sema: deleted `isMathBuiltin`/`isThreadBuiltin`/`isVectorCtorName` local name
    lists + the file-static `isThreadBuiltinName`; all three now forward to the
    Registry. Removed `isVectorCtorName` from Sema.h (no external callers).
  - GLSL: `isWarpIntrinsicName`/`isAtomicName`/`isAsyncCopyBuiltin` now check
    `builtinClass(name) == ...`. `lowerBuiltinCall`/`lowerAtomicName`/
    `glslVectorCtorName` (lowering) left in place.
  - MLIR: `isAtomicName` now checks `builtinClass`. `emitMathBuiltin`/
    `emitAtomicBuiltin`/`emitFenceBuiltin`/`emitVoteBuiltin`/`emitWarpBuiltin`/
    `emitAsyncCopyBuiltin` (lowering, inline name dispatch) left in place.
- Lowering logic NOT moved into the registry (GLSL produces strings, MLIR produces
  spirv ops — incompatible output forms).
- lit suite: 32/32 PASS. e2e: 68 PASS; the single FAIL is `printf.vc :: glsl` —
  pre-existing (undefined `vcEnableKernelPrintf`, unrelated).
- Builtin-heavy demos (atomics/builtins/geometric/half/vote/async_copy/features/
  sync) PASS on both backends.

