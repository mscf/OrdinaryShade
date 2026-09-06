"""Versioned shader export. No native handles or executable Python are serialized."""

from dataclasses import asdict
import hashlib

SCHEMA = "ordinaryshade/shader-v1"


def export_shader(shader):
    if shader.target != "wgsl" or shader.binary is not None:
        raise ValueError("Portable shaders require WGSL source")
    result = dict(
        schema=SCHEMA,
        target="wgsl",
        source=shader.source,
        sha256=hashlib.sha256(shader.source.encode()).hexdigest(),
        reflection=asdict(shader.reflection),
    )
    validate_shader_export(result)
    return result


def validate_shader_export(value):
    if value.get("schema") != SCHEMA or value.get("target") != "wgsl":
        raise ValueError("Unsupported shader export version/target")
    if not isinstance(value.get("source"), str) or not value["source"].strip():
        raise ValueError("Shader source is required")
    if hashlib.sha256(value["source"].encode()).hexdigest() != value.get("sha256"):
        raise ValueError("Shader source digest mismatch")
    reflection = value["reflection"]
    if (
        reflection["stage"] not in ("compute", "vertex", "fragment")
        or not reflection["entry_point"]
    ):
        raise ValueError("Invalid shader stage/entry point")
    bindings = [(r["set"], r["binding"]) for r in reflection["resources"]]
    if len(set(bindings)) != len(bindings) or any(a < 0 or b < 0 for a, b in bindings):
        raise ValueError("Duplicate or negative shader binding")
    return value
