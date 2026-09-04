# Roadmap

## Compiler foundation

- Fixed and runtime arrays, structures, scalar/vector/boolean-vector/matrix
  declarations, typed locals, arithmetic, calls, comparisons, conditionals,
  and resource indexing are established.
- Source spans, Python-to-generated-source mappings, and deterministic compiled
  content cache keys are implemented.
- A standalone versioned module serialization format remains future work.

## Compute language

- Typed helper composition, dynamic loops, mutable local/resource values,
  explicit casts, early returns, and statically bounded ranges are implemented.
- Structured storage/uniform buffers, storage images, sampled texture arrays,
  and Vulkan push constants with an explicit WGSL uniform fallback are implemented.
- Workgroup storage, barriers, atomics, subgroup compaction, Vulkan ray
  queries, and NVIDIA invocation reordering are implemented.
- Portable separate textures/samplers, depth comparison, 3-D sampling, and
  explicit texture LOD are implemented.
- Add specialization constants.

## Additional targets and stages

- Direct SPIR-V or Slang backend evaluation.
- Extend the implemented vertex/fragment stages and cross-stage location and
  builtin reflection with interpolation controls and additional render-target
  models.
- Ray-tracing stages only after cross-stage ABI rules are explicit.

## Ecosystem

- Maintain the existing Ordinary Light generated-shader compatibility gate
  without introducing a dependency from Ordinary Shade to the renderer.
- Support OrdinaryLattice through typed functions, resource reflection, and
  portable compute compilation; model-profile selection stays in that bridge.
- Shader package format, offline compilation, and stable reflection schema.
- Security model for applications that accept user-authored shaders.
