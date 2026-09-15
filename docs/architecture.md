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

## Fixed-size workgroup arrays

Compute functions and shader helpers can declare workgroup storage with
`values = osh.shared(osh.array(osh.u32, 64))`. The size must be a positive integer
literal. The compiler hoists this to `shared uint values[64]` in GLSL and
`var<workgroup> values: array<u32, 64>` in WGSL. Indexing has the element type;
scalar `osh.shared(osh.u32)` remains supported.

Initialize each element before reading it and use `osh.workgroup_barrier()`
between cooperative writes and reads. Every invocation in the workgroup must
reach that barrier, including lanes outside a partial final dispatch. Move
out-of-range early returns after the barrier. Storage is shared within a
workgroup, not across workgroups; a hoisted helper declaration is not a private
array per call. This facility does not add implicit barriers or initialization.
