# Ordinary Shade

Ordinary Shade is an experimental, typed Python shader language and compiler
toolkit. It is intended to compile a restricted Python subset through an owned
intermediate representation into GPU shader targets.

The first alpha supports Vulkan GLSL/SPIR-V and WebGPU WGSL compute, vertex,
and fragment shaders. It establishes the
compiler boundaries, diagnostics, resource reflection, and packaging needed to
add targets and shader stages without making GLSL the internal representation.
SPIR-V compilation currently uses the GLSL emitter and `glslangValidator`.

## Package organization

The Python package is organized into semantic namespaces: entry-point
declarations are in `ordinaryshade.entrypoints`, types and resources in
`ordinaryshade.types`, compiler IR in `ordinaryshade.ir`, compilation in
`ordinaryshade.compiler`, diagnostics and validation in their corresponding
namespaces, and target emitters in `ordinaryshade.targets`. The package-level
`ordinaryshade` imports remain the concise public API; the namespaces provide
stable, focused import paths for larger integrations.

Graphics interfaces use portable location and builtin semantics:

```python
import ordinaryshade as osh

@osh.vertex
def vertex_main(position: osh.location(osh.vec2, 0)) -> osh.builtin(osh.vec4, "position"):
    return osh.vec4(position, 0.0, 1.0)

@osh.fragment
def fragment_main() -> osh.location(osh.vec4, 0):
    return osh.vec4(0.9, 0.4, 0.1, 1.0)

vertex = osh.compile(vertex_main, target="spirv")
fragment = osh.compile(fragment_main, target="spirv")
pipeline = osh.link_graphics(vertex, fragment)
```

Compiled results include deterministic content cache keys and Python source
mappings for diagnostics and host-side shader/pipeline caches.

Graphics entry points may declare portable uniform buffers, storage buffers,
and storage records alongside location/builtin stage IO. Reflection exposes
the same descriptor set/binding model for GLSL/SPIR-V and WGSL so a host can
construct Vulkan descriptor sets or WebGPU bind groups from one declaration.

Reusable typed functions can also be generated for integration into a host
renderer's existing stages:

```python
@osh.function
def tint(color: osh.vec3, target: osh.vec3, strength: osh.f32) -> osh.vec3:
    return osh.mix(color, target, strength)

helper = osh.compile_function(tint)
```

The initial value vocabulary includes scalar, vector, boolean-vector, and
matrix types plus common constructors, arithmetic, relational expressions,
`power`, `clamp`, and vector-aware `select`. Ordinary Shade emits target-native
comparison and selection forms, including GLSL relational functions and WGSL
component-wise operators. Local values are explicitly typed in the IR;
structured conditionals, dynamic `while` loops, and statically bounded
`range()` loops lower to both targets. NumPy interoperability remains planned
rather than being inferred ambiguously.

```python
import ordinaryshade as osh


@osh.compute(workgroup_size=(8, 8, 1))
def copy_and_dim(
    source: osh.storage_image("rgba16f", access="read"),
    target: osh.storage_image("rgba16f", access="write"),
):
    pixel = osh.global_invocation_id.xy
    color = source.load(pixel)
    target.store(pixel, color * 0.5)


shader = osh.compile(copy_and_dim)
print(shader.source)
print(shader.reflection.resources)

webgpu_shader = osh.compile(copy_and_dim, target="wgsl")
print(webgpu_shader.source)
```

Portable structured resources use explicit declarations and retain stable
reflection names across targets:

```python
@osh.structure
class Sample:
    color: osh.vec3
    weight: osh.f32

@osh.structure
class Parameters:
    exposure: osh.f32

@osh.compute()
def process(
    samples: osh.storage_buffer(Sample, access="read"),
    params: osh.uniform_buffer(Parameters),
    target: osh.storage_image("rgba16f", access="write"),
):
    color = samples[0].color * samples[0].weight * params.exposure
    target.store(osh.ivec2(0), osh.vec4(color, 1.0))
```

`push_constants(Parameters)` is available for Vulkan GLSL and SPIR-V modules.
WGSL compilation rejects it unless an explicit uniform binding is supplied:
`push_constants(Parameters, wgsl_binding=7, wgsl_set=0)` emits a Vulkan
push-constant block and a WGSL uniform at group 0, binding 7. The host must bind
the WGSL uniform explicitly; reflection retains the `push_constants` kind and
records the supplied binding. Use `uniform_buffer` for a uniform on both targets.

Portable sampled resources include `sampled_texture_2d`,
`sampled_texture_3d`, and
`sampled_depth_texture_2d`, paired with `sampler` or `comparison_sampler`.
For example, a 3-D texture's `sample_level_with(sampler, uvw, lod)` lowers to
explicit-LOD sampling in GLSL and WGSL. Combined sampled-array resources and
other Vulkan-only facilities remain subject to target validation.

Storage-buffer elements and fields are writable with ordinary assignment.
Compute entry points support bare early returns and statically bounded Python
`range()` loops; dynamic or excessively large ranges are rejected during
lowering. This keeps loop bounds visible to validation and backend compilers.

Generated WGSL can be validated during compilation when the development-time
`naga-cli` tool is installed:

```bash
cargo install naga-cli --version 30.0.0 --locked
```

```python
webgpu_shader = osh.compile(copy_and_dim, target="wgsl", validate=True)
```

The validator is required by the WGSL CI job but remains optional for package
users and does not add a Rust dependency to Ordinary Shade at runtime.

This is not a compiler for arbitrary Python. Dynamic allocation, I/O,
exceptions, imports, reflection, recursion, and other Python runtime behavior
are intentionally unavailable in shader functions.

## Target-specific shader authoring

Complete production shaders may use explicitly declared backend capabilities
without embedding raw source. Ordinary Shade supports compute scheduling
builtins, workgroup-shared values and barriers, acceleration structures and ray
queries, subgroup ballot/compaction, atomics, and NVIDIA shader invocation
reordering. Unsupported Vulkan facilities are rejected by the WGSL backend.

Renderer-owned ABIs can be composed through `opaque_type()`, `inout()` function
parameters, `@external` function prototypes, and typed `external_values` passed
to `compile_function()`. These declarations do not redeclare the embedding
renderer's descriptor blocks; they make that ABI type-checkable while the
generated GLSL directly consumes it.

Workgroup-shared declarations are hoisted for both complete compute entry
points and reusable `compile_function()` output. This lets a renderer compose
Python-authored schedulers using shared storage, atomics, barriers, and compute
builtins into a larger backend-owned shader module.

## Project relationship

Ordinary Shade is renderer-independent and does not depend on Ordinary Light.
Ordinary Light already consumes its generated raster and wavefront shaders,
reflection, graphics programs, and compute modules. OrdinaryLattice separately
lowers LatticeModel graphs into Ordinary Shade functions and compute entries.
Both consumers own their integrations; neither is a dependency of this compiler.

## Development

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python -m build
```

The package is pre-alpha. The Python language subset and shader ABI may change
before the first stable release.
