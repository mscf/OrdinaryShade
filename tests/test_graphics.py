import shutil

import ordinaryshade as osh


@osh.structure
class VertexOutput:
    position: osh.builtin(osh.vec4, "position")
    color: osh.location(osh.vec3, 0)


@osh.vertex
def triangle_vertex(position: osh.location(osh.vec2, 0)) -> osh.builtin(osh.vec4, "position"):
    return osh.vec4(position, 0.0, 1.0)


@osh.fragment
def triangle_fragment() -> osh.location(osh.vec4, 0):
    return osh.vec4(0.95, 0.45, 0.15, 1.0)


@osh.vertex
def varying_vertex(position: osh.location(osh.vec2, 0)) -> VertexOutput:
    return VertexOutput(osh.vec4(position, 0.0, 1.0), osh.vec3(1.0, 0.5, 0.2))


@osh.fragment
def varying_fragment(color: osh.location(osh.vec3, 0)) -> osh.location(osh.vec4, 0):
    return osh.vec4(color, 1.0)


def test_glsl_graphics_stages_and_reflection():
    vertex = osh.compile(triangle_vertex)
    fragment = osh.compile(triangle_fragment)
    assert "layout(location = 0) in vec2 _osh_in_position;" in vertex.source
    assert "gl_Position = triangle_vertex(_osh_in_position);" in vertex.source
    assert "layout(location = 0) out vec4 _osh_out_result;" in fragment.source
    assert vertex.reflection.stage == "vertex"
    assert vertex.reflection.inputs[0].location == 0
    assert vertex.reflection.outputs[0].builtin == "position"


def test_wgsl_graphics_stages_validate_when_naga_is_available():
    validate = shutil.which("naga") is not None
    vertex = osh.compile(triangle_vertex, target="wgsl", validate=validate)
    fragment = osh.compile(triangle_fragment, target="wgsl", validate=validate)
    assert "@vertex" in vertex.source
    assert "@location(0) position: vec2<f32>" in vertex.source
    assert "-> @builtin(position) vec4<f32>" in vertex.source
    assert "@fragment" in fragment.source


def test_graphics_spirv_validates_when_glslang_is_available():
    if shutil.which("glslangValidator") is None:
        return
    assert osh.compile(triangle_vertex, target="spirv").binary[:4] == b"\x03\x02#\x07"
    assert osh.compile(triangle_fragment, target="spirv").binary[:4] == b"\x03\x02#\x07"


def test_cross_stage_varyings_link_for_both_targets():
    for target in ("glsl", "wgsl"):
        vertex = osh.compile(varying_vertex, target=target)
        fragment = osh.compile(varying_fragment, target=target)
        linked = osh.link_graphics(vertex, fragment)
        assert linked.varyings[0].type == "vec3"
        assert linked.varyings[0].location == 0
