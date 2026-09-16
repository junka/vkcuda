# Plan: P3A Function Overload Resolution

## 目标
把 Sema 的 `functions` 从 `name -> FunctionDecl*` 单映射改为支持同名重载，
实现 C/CUDA 风格的 overload resolution + conversion ranking，并给 device mangled
name 追加参数类型编码（让同名不同参的重载在后端符号表里区分开）。

这是 P3A 的本体。前置去重（mangling 5→1、builtin 列表 3→1）已完成。

## 现状（已直接读取确认）

### 数据结构
- `Sema::functions`：`llvm::StringMap<FunctionDecl*>`（Sema.h:42），单映射，后写覆盖。
- 后端 `funcDecls`/`funcTable`：也是单映射（GLSL ASTToGLSL.cpp:92, MLIR ASTToMLIR.cpp:641/744）。
- **关键问题**：两个同名重载 `f(int)` 和 `f(float)` 的 device mangled name 当前
  相同（`deviceMangledName` 只编码 namespace+name，不编码参数），后端 funcDecls 会
  互相覆盖。所以 P3A 必须同时做参数 mangle。

### 注册 / 查询点（全部已确认）
- 注册：`collectDecls` 一处（Sema.cpp:454 `functions[*fstore.back()] = f`）。
- 查询：两处——call 解析（Sema.cpp:995 `functions.find(calleeName)`）、launch 解析
  （Sema.cpp:1146 `functions.find(name)`）。
- 后端查询：GLSL funcDecls.find（1385, 1626）、MLIR funcTable.find（2681, 2767）+
  funcDecls.find（2696, 2775）。

### Ranking 积木（已存在，不需从零写）
- `isCompatibleForAssign(dst, src)`（Sema.cpp）：判参数兼容性。arithmetic↔arithmetic
  全兼容；pointer 要 exact pointee；同 kind 兼容。**这是 ranking 的基础**。
- `isArithmetic`/`isIntegerType`/`isFloatType`：类型分类。
- `resolveTypedefs`：剥 typedef。
- `checkCallArgs`（Sema.cpp:417）：已对单个 callee 做 per-arg 兼容检查 + float→int
  lossy 警告。**P3A 把这提升成 per-candidate 打分**。

## 设计决策（需确认）

### 决策 1：overload resolution 放 Sema 还是后端？

**方案 A**：Sema 做 resolution，在 CallExpr 上标注 `resolvedCallee`（新字段），
后端直接用，不重新选 overload。
- 优点：后端不改（funcDecls 仍可单映射，key 仍是 namespace-mangled name，因为
  Sema 已选定唯一 callee）。
- 缺点：CallExpr 要加字段；Sema 要存完整 overload set；host 后端不需要 Sema 的
  resolution（host 是 C++ 原生 overload）。

**方案 B**：Sema 只做诊断（ambiguous/no viable 报错），后端用**参数 mangled name**
自己查表（`f(int)`→`f_i`、`f(float)`→`f_f`，天然区分）。
- 优点：CallExpr 不改；后端 funcDecls key 改成参数 mangled name 即可。
- 缺点：后端要重算参数 mangled name（需 Sema 推断出的参数类型）；call site 的
  arg 类型在后端 emit 时不一定都已知（GLSL 后端不做完整类型推断）。

**推荐方案 C（混合）**：Sema 做 resolution 并在 CallExpr 标注 `resolvedCallee`
（方案 A），**同时**给 `deviceMangledName` 加参数编码（让后端 emit 的符号区分开，
因为两个重载 `f(int)`/`f(float)` 要 emit 成不同符号 `f_i`/`f_f`，否则 GLSL/SPIR-V
里两个同名函数定义冲突）。后端 emit 函数定义时用参数 mangled name，emit call 时
用 Sema 标注的 resolvedCallee 的 mangled name。

这样：
- Sema：存 overload set（`StringMap<SmallVector<FunctionDecl*>>`），做 resolution，
  CallExpr 标注 resolvedCallee。
- 后端：funcDecls key 改成参数 mangled name（`f_i`/`f_f`），emit 定义和 call 都用它。
- 参数 mangle：`deviceMangledName(f)` 末尾追加参数类型编码。

### 决策 2：参数 mangle 编码格式

简单方案（不追求 Itanium ABI 兼容，只求唯一可读）：
```
f(int)            -> f_i
f(float)          -> f_f
f(int*, float)    -> f_pi_f
f(float4)         -> f_f4
f()               -> f_v (void/empty)
Class::method(int)-> Class_method_i
ns::f(int, float) -> ns_f_i_f
```
类型编码：`i`=int32, `u`=uint32, `I`=int64, `U`=uint64, `f`=float32, `d`=float64,
`h`=float16, `b`=bool, `v`=void, `p`=pointer(后跟 pointee), `V<n><elem>`=vector。

**风险**：改了 device 符号名，所有现有 demo 的 kernel symbol 变了（`fill`→`fill_pi`），
host 的 `__vc_spirv_<kname>` 和 entry point 也要同步。但 host 的 `launchHandleName`
读的是 call site 的 callee（ DeclRefExpr name 或 scope chain），**不含参数**——
所以 kernel launch handle 仍是 `fill`，而 MLIR emit 的 func.func 是 `fill_pi`，
**两者会不匹配**。

这是决策 2 的最大风险：**kernel 的 launch handle（host 端，无参数）必须和 MLIR 的
func.func symbol（含参数）一致**。两个选择：
- (a) kernel 不 mangle 参数（kernel 不可重载，CUDA 里 `__global__` 不支持 overload），
  只对 `__device__` 函数 mangle 参数。
- (b) 全部 mangle，host launchHandleName 也改成查 Sema resolvedCallee。

**(a) 更安全**：CUDA 语义里 `__global__` 函数本就不允许 overload（launch 语法
`kernel<<<...>>>(args)` 无法区分重载），所以 kernel 保持无参数 mangle，只 `__device__`
函数加参数 mangle。这样 kernel symbol 不变，host launch 路径零改动。

### 决策 3：conversion ranking 打分

C/CUDA 等级（`checkCallArgs` 现有逻辑提升成打分）：
```
Rank (越小越好):
  0  exact match          (arg type == param type, typedefs resolved)
  1  promotion            (int→int/long, float→double — 无损提升)
  2  standard conversion  (int→float, float→int, 同 kind 不同宽度)
  3  lossy                (float→int, 大→小窄化)
  4  incompatible         (pointer mismatch, kind mismatch)
  ∞  arity mismatch       (arg count != param count, defaults applied)
```
- 每个 candidate 取所有 arg 的**最差 rank**作为 candidate rank。
- 选 rank 最小的 candidate；并列最小→ambiguous（报候选 notes）。
- 无 viable candidate（都 ∞）→ no viable（报候选 notes）。

## 实施步骤（按决策 C + 2(a) + 3）

### Step 1: CallExpr 加 resolvedCallee 字段
- AST.h CallExpr 加 `FunctionDecl *resolvedCallee = nullptr;`（Sema 填，后端读）。
- 不 owned（FunctionDecl 归 TranslationUnit）。

### Step 2: deviceMangledName 加参数编码（仅 __device__，kernel 不加）
- Mangle.h/cpp：`deviceMangledName(f)` 末尾追加 `_<param encodings>`，但
  `f->deviceAttr == Global` 时不加（保持 kernel symbol = namespace_name）。
- 加一个参数类型编码 helper `mangleType(const Type*)`。
- **这是破坏性改动**：所有 `__device__` 函数符号名变了。但 kernel 不变，所以 host
  launch 路径不受影响。后端 funcDecls key 自动区分重载。

### Step 3: Sema functions 改 overload set
- `functions`：`StringMap<FunctionDecl*>` → `StringMap<SmallVector<FunctionDecl*>>`。
  key 仍是 namespace-mangled name（不含参数），value 是同 name 的所有 decl。
- 注册（collectDecls:454）：push_back 而非覆盖。同签名重复定义仍报 redefinition。

### Step 4: Sema overload resolution
- call 解析（Sema.cpp:995）：取 overload set，对每个 candidate 算 rank，选 best。
  - 填 CallExpr::resolvedCallee。
  - ambiguous/no viable 报错（带候选 notes——但 Note 机制还没有，先报 error 列候选）。
  - 选定后仍调 checkCallArgs（保留 lossy 警告等）。
- launch 解析（Sema.cpp:1146）：kernel 不可重载，取 set 第一个（或唯一）Global。

### Step 5: 后端用 resolvedCallee + 参数 mangled name
- GLSL/MLIR funcDecls key 改成 `deviceMangledName(f)`（现在含参数编码，自动区分）。
- emit call：用 `c->resolvedCallee`（若 Sema 填了）的 mangled name，否则 fallback
  到原 name 查表（向后兼容 builtins/未解析的）。
- emit 定义：用 `deviceMangledName(f)`。

### Step 6: 测试
- 新增 `test/Frontend/overload-*.vc`：exact match、ambiguous、no viable、default args
  与 overload 交互、builtin overload（sinf vs sin）。
- 新增 `test/MLIR/overload-*.vc`：FileCheck 参数 mangled symbol。
- e2e：现有 demo 不回归（`__device__` 符号变了，但 call site 也用新 mangled name）。

## 风险

1. **`__device__` 符号名变更是破坏性的**：所有现有 demo 的 device helper 符号从
   `helper` 变 `helper_i` 之类。但只要 call site 和定义用同一个 `deviceMangledName`，
   就自洽。风险在 host 端：host 不 mangle 参数，但 host 调 device 是通过 launch
   handle（kernel，无参数 mangle），不直接 call `__device__` 函数，所以 host 不受影响。
2. **kernel 不 mangle 参数是关键约束**：若误给 kernel 加参数编码，host launchHandleName
   （无参数）和 MLIR func.func symbol（含参数）不匹配，launch 失败。Step 2 必须对
   Global 排除。
3. **builtin overload**：`sin`/`sinf`/`__sinf` 当前是不同名字（不是 overload），Sema
   不 resolve 它们（走 isBuiltin pass-through）。P3A 不改 builtin 路径——builtin
   overload 是后续（用 BuiltinRegistry 加签名）。
4. **redefinition 检查**：改 overload set 后，`f(int)` 定义两次仍要报错（同签名），
   但 `f(int)` 和 `f(float)` 合法。需要加签名比较（Type equality——但 Type 无
   canonical/equality，这是 P3A 的另一个子问题）。

## Type equality 子问题
P3A 需要"两个函数签名是否相同"判断（redefinition + overload set 去重）。但 Type
类无 equality/hash（规划 P3A 的"Canonical Type"目标）。短期方案：写一个
`sameType(a, b)` 结构比较（resolveTypedefs 后比 kind + 子字段），不引入 canonical。
完整 canonical 留给后续。

## 不做
- 不做 builtin overload（sin/sinf 仍按名字区分，走 Registry pass-through）。
- 不做完整 canonical Type / equality / hash（用结构比较 sameType 代替）。
- 不做 host 端 overload（host 是 C++ 原生，g++ 解析）。
- 不改 kernel mangle（kernel 保持 namespace_name，无参数编码）。

## 交付物
- AST.h：CallExpr 加 resolvedCallee。
- Mangle.h/cpp：deviceMangledName 加参数编码（Global 排除）+ mangleType helper。
- Sema.h/cpp：functions 改 overload set，加 overload resolution + ranking + sameType。
- ASTToGLSL.cpp / ASTToMLIR.cpp：funcDecls key + call emit 用 resolvedCallee。
- 测试：test/Frontend/overload-*.vc、test/MLIR/overload-*.vc。
- 验证：lit + e2e 全过（除预存在失败）。
