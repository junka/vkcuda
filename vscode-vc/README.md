# VC — VSCode Language Support

Syntax highlighting for the **VC** language — a CUDA-like language targeting
Vulkan via MLIR/SPIR-V (see the parent [`gg`](..) project).

This extension provides TextMate-grammar-based syntax highlighting for `.vc`
files. It is pure-declarative (no language server, no compiled code) so it works
in any editor that loads TextMate grammars (VS Code, VSCodium, Cursor, etc.).

## What gets highlighted

| Category | Examples |
|---|---|
| Control flow | `if else for while do return break continue switch case default goto` |
| Type / storage keywords | `void bool int unsigned uint long float double half __half struct enum class namespace typedef const static extern` |
| CUDA execution-space | `__global__ __device__ __host__ __shared__ __constant__ __restrict__` |
| Vector types | `float4 int3 uint2 double2 long3 bool4 half2 half3 half4` (+ `make_float4`) |
| Thread-index builtins | `threadIdx blockIdx blockDim gridDim warpSize` |
| Sync / fence | `__syncthreads __threadfence __syncthreads_count __syncwarp` |
| Warp / subgroup | `__shfl_sync __shfl_down_sync __ballot_sync __anySync __allSync __activemask` |
| Atomics | `atomicAdd atomicSub atomicExch atomicCAS atomicMin atomicMax atomicAnd ...` |
| Math (GLSL + f-suffix + fast intrinsics) | `sin cos dot cross normalize clamp mix fma` · `sinf cosf` · `__sinf __cosf __expf` |
| VC runtime API | `vcInit vcMalloc vcMemcpy vcLaunchKernel vcDeviceSynchronize vcMemcpyPeer ...` |
| Enum constants | `VCError::Success`, `VCMemcpyKind::DeviceToHost` |
| Kernel launch | `kernel<<<grid, block, stream>>>(args)` |
| Literals | hex `0xffffffff`, float `1.5e-3f`, `1.0f`, char/string w/ escapes |
| Preprocessor | `#include`, `#define NAME <literal>`, `#ifdef/#endif` |
| Comments | `//` line, `/* */` block, `/** */` doc-blocks (with `@param`/`@return`) |

## Install

### Option A — install as a local VSIX (no build step)

The grammar ships as plain JSON, so you can package and install it with the
`vsce` CLI (the only dependency):

```bash
cd vscode-vc
npx @vscode/vsce package      # produces vc-0.1.0.vsix
code --install-extension vc-0.1.0.vsix
```

Then reload the window. `.vc` files now highlight.

### Option B — run from source (Extension Development Host)

1. In VS Code, open the `vscode-vc/` folder.
2. Add a launch config (or press **F5** → "Extension Development Host").
3. A new VS Code window opens with the VC grammar loaded. Open any `.vc` file
   in the parent `test/` directory to see it in action.

### Option C — copy into your user extensions dir

```bash
cp -r vscode-vc ~/.vscode/extensions/vc-0.1.0
```

(restart VS Code afterwards)

## Project layout

```
vscode-vc/
├── package.json                  # contribution points: language id + grammar
├── language-configuration.json   # bracket autoclose, comment toggling, indent
├── syntaxes/
│   └── vc.tmLanguage.json        # the TextMate grammar (the core of this ext)
├── README.md
└── example/
    └── highlight-demo.vc         # exercises every token category above
```

## Token scope reference

Themes color scopes following [TextMate conventions](https://macromates.com/manual/en/language_grammars).
This grammar targets standard scopes so any theme works out of the box:

- `keyword.control.vc`, `storage.type.builtin.vc`, `storage.modifier.cuda.vc`
- `support.type.vector.vc`, `support.type.enum.vc`
- `support.variable.cuda.vc`, `support.function.builtin.*.vc`
- `support.function.vc-runtime.vc`, `constant.other.enummember.vc`
- `constant.numeric.{integer,float}.*.vc`, `constant.language.boolean.*.vc`
- `string.quoted.{double,single}.vc`, `comment.{line,block}.*.vc`
- `meta.preprocessor.{include,define,conditional}.vc`
- `meta.function-call.launch.vc`, `entity.name.function.vc`

## Updating the keyword lists

The grammar mirrors the VC lexer's keyword tables at
[`include/vc/Frontend/Lexer.h`](../include/vc/Frontend/Lexer.h) and the builtin
lists in [`lib/Frontend/Sema.cpp`](../lib/Frontend/Sema.cpp) and
[`include/vc/Runtime/VCRuntime.h`](../include/vc/Runtime/VCRuntime.h). If a new
builtin or runtime call is added upstream, add a `|`-separated alternative to
the relevant `match` in [`syntaxes/vc.tmLanguage.json`](syntaxes/vc.tmLanguage.json).

## License

MIT.
