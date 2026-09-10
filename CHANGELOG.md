# Changelog

## 0.1.0a5

- Support typed compile-time shader variants, output parameters, runtime-array lengths, and 3D storage images.
- Add math, packing, and Vulkan atomic intrinsics required by OrdinaryLight transport.
- Fix vector bitcasts, loop-local type scoping, graphics helper resource access, and resource-free helper structure registration.
- Extend external resource contexts and Vulkan image formats; reject unsupported WGSL operations explicitly.
- Validate the complete OrdinaryLight shader migration with 106 compiler tests.

## 0.1.0a0

- Establish the independent `ordinaryshade` package and compiler architecture.
- Add compute entry points, storage-image declarations, typed IR, GLSL output,
  reflection metadata, diagnostics, and optional glslang SPIR-V compilation.
- Add WGSL output with optional Naga validation.
- Add vertex and fragment stages for GLSL, WGSL, and stage-aware SPIR-V,
  including structured outputs and cross-stage interface validation.
- Add deterministic compiled-module cache keys and Python source mappings.
- Add structured buffers, uniform buffers, push constants, fixed/runtime/local
  arrays, sampled texture arrays, storage image arrays, and typed reflection.
- Add helper composition, typed external ABIs, inout parameters, backend-owned
  external values, dynamic and bounded loops, workgroup storage, barriers,
  atomics, subgroup operations, Vulkan ray queries, and shader invocation
  reordering.
- Validate the compiler against Ordinary Light's complete generated path-tracing
  shader suite.
