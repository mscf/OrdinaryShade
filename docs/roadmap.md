# Roadmap

## Compiler foundation

- Fixed and runtime arrays, structures, scalar/vector/boolean-vector/matrix
  declarations, typed locals, arithmetic, calls, comparisons, conditionals,
  and resource indexing are established.
- Add source spans and Python-to-generated-source diagnostic mapping.
- Define deterministic module serialization and cache keys.

## Compute language

- Typed helper composition, dynamic loops, mutable local/resource values,
  explicit casts, early returns, and statically bounded ranges are implemented.
- Structured storage/uniform buffers, storage images, sampled texture arrays,
  and Vulkan-only push constants are implemented.
- Workgroup storage, barriers, atomics, subgroup compaction, Vulkan ray
  queries, and NVIDIA invocation reordering are implemented.
- Add specialization constants and a portable sampled texture/sampler model.

## Additional targets and stages

- Direct SPIR-V or Slang backend evaluation.
- Extend the implemented vertex/fragment stages and cross-stage location and
  builtin reflection with interpolation controls and additional render-target
  models.
- Ray-tracing stages only after cross-stage ABI rules are explicit.

## Ecosystem

- Maintain an Ordinary Light compatibility gate without introducing a runtime
  dependency from Ordinary Shade to the renderer.
- Shader package format, offline compilation, and stable reflection schema.
- Security model for applications that accept user-authored shaders.
