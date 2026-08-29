# Architecture

The compiler pipeline is intentionally layered:

```text
restricted Python AST
        ↓
Ordinary Shade typed IR
        ├── GLSL emitter → glslang → SPIR-V
        └── WGSL emitter → WebGPU shader module
                    + portable reflection
```

The Python AST is never executed as shader code. Decorated functions retain
their source for lowering, and unsupported syntax is rejected before backend
emission.

GLSL and WGSL are backends rather than internal languages. Future emitters may
target Slang or SPIR-V without changing the user-facing Python subset.

Ordinary Shade has no renderer dependency. Renderer integrations consume its
compiled modules and reflection through adapters owned by those renderers.
