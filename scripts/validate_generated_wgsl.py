"""Validate representative Ordinary Shade WGSL output with Naga."""

from __future__ import annotations

import ordinaryshade as osh


@osh.compute(workgroup_size=(8, 8, 1))
def copy_image(
    source: osh.storage_image("rgba16f", access="read"),
    target: osh.storage_image("rgba16f", access="write"),
):
    pixel = osh.global_invocation_id.xy
    color = source.load(pixel)
    target.store(pixel, color * 0.5)


@osh.function
def tint(color: osh.vec3, target: osh.vec3, strength: osh.f32) -> osh.vec3:
    return osh.mix(color, target, strength)


@osh.function
def choose(condition: osh.boolean, yes: osh.vec3, no: osh.vec3) -> osh.vec3:
    return osh.select(condition, yes, no)


def main() -> None:
    result = osh.compile(copy_image, target="wgsl", validate=True)
    osh.compile_function(tint, target="wgsl", validate=True)
    osh.compile_function(choose, target="wgsl", validate=True)
    print(
        "Validated generated WGSL compute and helper modules with Naga: "
        f"{result.reflection.entry_point} {result.reflection.workgroup_size}"
    )


if __name__ == "__main__":
    main()
