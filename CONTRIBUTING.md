# Contributing

Run the hardware-independent test suite before submitting a change:

```bash
python -m unittest discover -s tests -v
```

Compiler changes should include positive and negative language tests. Public
language or ABI changes must update the README, changelog, and relevant design
document.

