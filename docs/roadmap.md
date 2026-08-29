# Roadmap

## Compiler foundation

- Add fixed-array types. Structures plus scalar, vector, boolean-vector, and
  matrix value declarations are established.
- Extend the typed expression pass beyond the implemented parameters, locals,
  arithmetic, calls, comparisons, and conditionals to structures and arrays.
- Add source spans and Python-to-generated-source diagnostic mapping.
- Define deterministic module serialization and cache keys.

## Compute language

- Expand the initial typed helper functions with cross-function calls.
- Dynamic loops and mutable local variables. Structured `if`/`else`, immutable
  typed locals, explicit casts, resource indexing, writable storage elements,
  early returns, and statically bounded `range()` loops are implemented.
- Sampled images and samplers. Structured storage/uniform buffers are
  implemented, along with Vulkan-only push constants and portable reflection.
- Shared memory, barriers, atomics, specialization constants, and subgroups.

## Additional targets and stages

- Direct SPIR-V or Slang backend evaluation.
- Vertex and fragment stages after compute semantics stabilize.
- Ray-tracing stages only after cross-stage ABI rules are explicit.

## Ecosystem

- Ordinary Light adapter as an optional Ordinary Light dependency.
- Shader package format, offline compilation, and stable reflection schema.
- Security model for applications that accept user-authored shaders.
