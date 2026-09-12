# Changelog

## 0.1.0a7

- Add typed signed/unsigned integer compare-and-exchange for GLSL/SPIR-V.
- Validate matching operands and explicitly reject unsupported WGSL use.
- Exercise GPU hash-table use through vxl8r visible-face averaging.

## 0.1.0a6

- Add typed integer `atomic_exchange` for GLSL/SPIR-V; explicitly reject unsupported WGSL use.
- Correct ray-query instance identity handling for application-owned acceleration structures.
- Validate with 107 compiler tests (one optional test skipped) and native GPU integration tests.

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
