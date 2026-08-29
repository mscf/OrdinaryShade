# Ordinary Shade

Ordinary Shade is an experimental, typed Python shader language and compiler
toolkit. It is intended to compile a restricted Python subset through an owned
intermediate representation into GPU shader targets.

The first alpha supports Vulkan GLSL/SPIR-V and WebGPU WGSL compute, vertex,
and fragment shaders. It establishes the
compiler boundaries, diagnostics, resource reflection, and packaging needed to
grow toward SPIR-V and additional shader stages without making GLSL the
project's internal representation.

Graphics interfaces use portable location and builtin semantics:

```python
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
reflection names across backends:

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
WGSL compilation rejects it explicitly because WebGPU has no push-constant
resource; portable modules should use `uniform_buffer` instead.

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

## Backend-specific shader authoring

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
Ordinary Light may later provide an optional adapter or package extra that
accepts Ordinary Shade modules at stable renderer extension points.

## Development

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python -m build
```

The package is pre-alpha. The Python language subset and shader ABI may change
before the first stable release.
