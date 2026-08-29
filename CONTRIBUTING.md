# Contributing

Run the hardware-independent test suite before submitting a change:

```bash
python -m unittest discover -s tests -v
```

The test suite requires `glslangValidator` for GLSL/SPIR-V validation. On
Ubuntu it is provided by `glslang-tools`. WGSL validation additionally uses
`naga-cli`; CI runs both validators.

Compiler changes should include positive and negative language tests. Public
language or ABI changes must update the README, changelog, and relevant design
document.
