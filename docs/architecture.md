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

Compute and graphics compilation share typed resource declarations and portable
set/binding reflection. Vertex/fragment linking validates stage interfaces.
Compiled results include deterministic content keys and Python source mappings;
host integrations own pipeline caches and resource lifetimes.

Portable separate texture/sampler declarations lower to both targets. Vulkan
capabilities such as ray queries, subgroup facilities, and invocation reordering
are explicit and rejected by WGSL when unsupported. A push-constant declaration
may opt into a named WGSL uniform binding; this is a host-visible resource
contract, not an implicit WebGPU push-constant feature.
