import unittest
import shutil

import ordinaryshade as osh


@osh.structure
class ColorRecord:
    color: osh.vec3
    weight: osh.f32


@osh.structure
class ScaleParameters:
    scale: osh.f32


@osh.structure
class ClearParameters:
    count: osh.u32


@osh.structure
class QueueState:
    count: osh.u32
    capacity: osh.u32
    overflow: osh.u32
    padding: osh.u32


@osh.structure
class DispatchState:
    x: osh.u32
    y: osh.u32
    z: osh.u32


@osh.compute(workgroup_size=(8, 4, 1))
def copy_image(
    source: osh.storage_image("rgba16f", access="read"),
    target: osh.storage_image("rgba16f", access="write"),
):
    pixel = osh.global_invocation_id.xy
    color = source.load(pixel)
    target.store(pixel, color * 0.5)


class CompilerTests(unittest.TestCase):
    def test_math_intrinsics_and_loop_continue_validate(self):
        @osh.function
        def weight(value: osh.f32) -> osh.f32:
            return osh.exp(-osh.absolute(value)) * osh.sqrt(value * value)

        @osh.compute()
        def filter_words(words: osh.storage_buffer(osh.f32)):
            for index in range(4):
                if index == 2:
                    continue
                words[index] = weight(words[index])

        glsl = osh.compile(filter_words, helpers=(weight,))
        self.assertIn("exp((-abs(value)))", glsl.source)
        self.assertIn("continue;", glsl.source)
        wgsl = osh.compile(
            filter_words, helpers=(weight,), target="wgsl", validate=True,
        )
        self.assertIn("exp((-abs(value)))", wgsl.source)
        self.assertIn("continue;", wgsl.source)

    def test_typed_unary_operators_validate(self):
        @osh.function
        def signed(value: osh.f32, enabled: osh.boolean) -> osh.f32:
            if not enabled:
                return -value
            return +value

        glsl = osh.compile_function(signed)
        self.assertIn("if ((!enabled))", glsl.source)
        self.assertIn("return (-value);", glsl.source)
        wgsl = osh.compile_function(signed, target="wgsl", validate=True)
        self.assertIn("if ((!enabled))", wgsl.source)

    def test_composed_helpers_can_call_other_helpers(self):
        @osh.function
        def quantize(value: osh.f32) -> osh.u32:
            return osh.u32(osh.round(value))

        @osh.function
        def shifted(value: osh.f32) -> osh.u32:
            return quantize(value) << osh.u32(8)

        @osh.compute()
        def write(words: osh.storage_buffer(osh.u32, access="write")):
            words[0] = shifted(1.0)

        glsl = osh.compile(write, helpers=(quantize, shifted))
        self.assertIn("return (quantize(value) << uint(8))", glsl.source)
        wgsl = osh.compile(
            write, helpers=(quantize, shifted), target="wgsl", validate=True,
        )
        self.assertIn("return (quantize(value) << u32(8))", wgsl.source)

    def test_round_shift_and_bitwise_composition_validate(self):
        @osh.function
        def pack_bytes(a: osh.u32, b: osh.u32, value: osh.f32) -> osh.u32:
            rounded = osh.u32(osh.round(value))
            return a | (b << osh.u32(8)) | (rounded << osh.u32(16))

        glsl = osh.compile_function(pack_bytes)
        self.assertIn("uint(round(value))", glsl.source)
        self.assertIn("b << uint(8)", glsl.source)
        wgsl = osh.compile_function(pack_bytes, target="wgsl", validate=True)
        self.assertIn("u32(round(value))", wgsl.source)
        self.assertIn("b << u32(8)", wgsl.source)

    def test_image_size_and_dynamic_nested_ranges_validate(self):
        @osh.compute()
        def fill(image: osh.storage_image("rgba8", access="write", binding=0)):
            extent = image.size()
            for y in range(osh.i32(0), extent.y):
                for x in range(osh.i32(0), extent.x):
                    image.store(osh.ivec2(x, y), osh.vec4(1.0))

        glsl = osh.compile(fill)
        self.assertIn("ivec2 extent = imageSize(image);", glsl.source)
        self.assertIn("for (int y = int(0); y < extent.y; y += 1)", glsl.source)
        wgsl = osh.compile(fill, target="wgsl", validate=True)
        self.assertIn("vec2<i32>(textureDimensions(image))", wgsl.source)
        self.assertIn("for (var x: i32 = i32(0);", wgsl.source)

    def test_compute_entrypoint_composes_typed_helpers(self):
        @osh.function
        def brighten(color: osh.vec3) -> osh.vec3:
            return osh.minimum(color * 2.0, osh.vec3(1.0))

        @osh.compute()
        def packed(
            records: osh.storage_buffer(ColorRecord, access="read", binding=0),
            words: osh.storage_buffer(osh.u32, access="write", binding=1),
        ):
            color = brighten(records[0].color)
            words[0] = osh.pack_unorm4x8(osh.vec4(color, 1.0))

        glsl = osh.compile(packed, helpers=(brighten,))
        self.assertIn("vec3 brighten(vec3 color)", glsl.source)
        self.assertIn("packUnorm4x8(vec4(color, 1.0))", glsl.source)
        wgsl = osh.compile(
            packed, helpers=(brighten,), target="wgsl", validate=True,
        )
        self.assertIn("fn brighten(color: vec3<f32>)", wgsl.source)
        self.assertIn("pack4x8unorm(vec4<f32>(color, 1.0))", wgsl.source)

    def test_integer_vector_constructors_and_swizzles_validate(self):
        @osh.function
        def repack(value: osh.uvec4) -> osh.uvec4:
            return osh.uvec4(value.xy, osh.u32(value.z), value.w)

        glsl = osh.compile_function(repack)
        self.assertIn("uvec4(value.xy, uint(value.z), value.w)", glsl.source)
        wgsl = osh.compile_function(repack, target="wgsl", validate=True)
        self.assertIn("vec4<u32>(value.xy, u32(value.z), value.w)", wgsl.source)

    def test_write_only_structured_buffer_preserves_target_access(self):
        @osh.compute()
        def write_record(
            records: osh.storage_buffer(ColorRecord, access="write", binding=0),
        ):
            records[0].color = osh.vec3(1.0)
            records[0].weight = osh.f32(1.0)

        glsl = osh.compile(write_record)
        self.assertIn("writeonly buffer records_Block", glsl.source)
        wgsl = osh.compile(write_record, target="wgsl", validate=True)
        self.assertIn(
            "var<storage, read_write> records: array<ColorRecord>;",
            wgsl.source,
        )
        if shutil.which("glslangValidator"):
            self.assertGreater(
                len(osh.compile(write_record, target="spirv").binary), 20
            )

    def test_fixed_storage_records_emit_and_validate_for_both_targets(self):
        @osh.compute()
        def prepare(
            queue: osh.storage_record(QueueState, access="read", binding=0),
            dispatch: osh.storage_record(DispatchState, access="write", binding=1),
        ):
            active_count = osh.minimum(queue.count, queue.capacity)
            dispatch.x = (active_count + osh.u32(63)) / osh.u32(64)
            dispatch.y = osh.u32(1)
            dispatch.z = osh.u32(1)

        glsl = osh.compile(prepare)
        self.assertIn("readonly buffer queue_Block", glsl.source)
        self.assertIn("writeonly buffer dispatch_Block", glsl.source)
        self.assertIn("uint count;", glsl.source)
        self.assertIn(
            "dispatch.x = ((active_count + uint(63)) / uint(64));",
            glsl.source,
        )
        self.assertEqual(
            [item.format for item in glsl.reflection.resources],
            ["QueueState", "DispatchState"],
        )
        wgsl = osh.compile(prepare, target="wgsl", validate=True)
        self.assertIn("var<storage, read> queue: QueueState;", wgsl.source)
        self.assertIn("var<storage, read_write> dispatch: DispatchState;", wgsl.source)
        if shutil.which("glslangValidator"):
            self.assertGreater(len(osh.compile(prepare, target="spirv").binary), 20)

    def test_writable_scalar_buffer_early_return_and_bounded_loop(self):
        @osh.compute(workgroup_size=(64, 1, 1))
        def clear_words(
            words: osh.storage_buffer(osh.u32, binding=0),
            params: osh.uniform_buffer(ClearParameters, binding=1),
        ):
            item = osh.global_invocation_id.x
            if item >= params.count:
                return
            base = item * osh.u32(4)
            for offset in range(4):
                words[base + osh.u32(offset)] = osh.u32(0)

        glsl = osh.compile(clear_words)
        self.assertIn("uint words[];", glsl.source)
        self.assertIn("return;", glsl.source)
        self.assertIn("for (int offset = 0; offset < 4; offset += 1)", glsl.source)
        if shutil.which("glslangValidator"):
            self.assertGreater(len(osh.compile(clear_words, target="spirv").binary), 20)
        wgsl = osh.compile(clear_words, target="wgsl", validate=True)
        self.assertIn("var<storage, read_write> words: array<u32>;", wgsl.source)
        self.assertIn("for (var offset: i32 = 0;", wgsl.source)

    def test_structured_buffers_emit_and_validate_for_both_targets(self):
        @osh.compute()
        def buffered(
            records: osh.storage_buffer(ColorRecord, access="read"),
            params: osh.uniform_buffer(ScaleParameters),
            target: osh.storage_image("rgba16f", access="write"),
        ):
            weighted = records[0].color * records[0].weight
            output = osh.vec4(weighted * params.scale, 1.0)
            target.store(osh.ivec2(0), output)

        glsl = osh.compile(buffered)
        self.assertIn("struct ColorRecord", glsl.source)
        self.assertIn("readonly buffer records_Block", glsl.source)
        self.assertIn("uniform params_Block", glsl.source)
        self.assertIn("vec3 weighted =", glsl.source)
        self.assertEqual(
            [item.kind for item in glsl.reflection.resources],
            ["storage_buffer", "uniform_buffer", "storage_image"],
        )
        wgsl = osh.compile(buffered, target="wgsl", validate=True)
        self.assertIn("var<storage, read> records: array<ColorRecord>;", wgsl.source)
        self.assertIn("var<uniform> params: ScaleParameters;", wgsl.source)
        if shutil.which("glslangValidator"):
            spirv = osh.compile(buffered, target="spirv")
            self.assertGreater(len(spirv.binary), 20)

    def test_push_constants_are_explicitly_vulkan_only(self):
        @osh.compute()
        def pushed(
            params: osh.push_constants(ScaleParameters),
            target: osh.storage_image("rgba16f", access="write"),
        ):
            target.store(osh.ivec2(0), osh.vec4(params.scale))

        glsl = osh.compile(pushed)
        self.assertIn("layout(push_constant) uniform params_Block", glsl.source)
        self.assertEqual(glsl.reflection.resources[0].kind, "push_constants")
        with self.assertRaisesRegex(osh.ShaderTypeError, "no push-constant"):
            osh.compile(pushed, target="wgsl")

    def test_explicit_scalar_casts_are_target_native(self):
        @osh.function
        def cast_value(value: osh.u32) -> osh.f32:
            return osh.f32(value)

        self.assertIn("return float(value);", osh.compile_function(cast_value).source)
        wgsl = osh.compile_function(cast_value, target="wgsl", validate=True)
        self.assertIn("return f32(value);", wgsl.source)

    def test_typed_helper_function_emits_glsl(self):
        @osh.function
        def tint(color: osh.vec3, target: osh.vec3, strength: osh.f32) -> osh.vec3:
            return osh.mix(color, target, strength)

        result = osh.compile_function(tint)
        self.assertEqual(result.name, "tint")
        self.assertIn(
            "vec3 tint(vec3 color, vec3 target, float strength)", result.source,
        )
        self.assertIn("return mix(color, target, strength);", result.source)

    def test_helper_intrinsics_and_vector_constructors(self):
        @osh.function
        def highlight(color: osh.vec3, tint: osh.vec3, strength: osh.f32) -> osh.vec3:
            return osh.minimum(color + tint * strength, osh.vec3(1.0))

        result = osh.compile_function(highlight)
        self.assertIn(
            "return min((color + (tint * strength)), vec3(1.0));",
            result.source,
        )

    def test_clamp_intrinsic_emits_for_glsl_and_wgsl(self):
        @osh.function
        def aces(color: osh.vec3) -> osh.vec3:
            return osh.clamp(
                (color * (2.51 * color + 0.03))
                / (color * (2.43 * color + 0.59) + 0.14),
                osh.vec3(0.0),
                osh.vec3(1.0),
            )

        glsl = osh.compile_function(aces)
        self.assertIn("return clamp(", glsl.source)
        self.assertIn("vec3(0.0), vec3(1.0)", glsl.source)
        wgsl = osh.compile_function(aces, target="wgsl", validate=True)
        self.assertIn("return clamp(", wgsl.source)
        self.assertIn("vec3<f32>(0.0), vec3<f32>(1.0)", wgsl.source)

    def test_select_lowers_to_typed_ternary(self):
        @osh.function
        def choose(condition: osh.boolean, yes: osh.vec3, no: osh.vec3) -> osh.vec3:
            return osh.select(condition, yes, no)

        result = osh.compile_function(choose)
        self.assertIn("return (condition ? yes : no);", result.source)

    def test_vector_comparison_power_and_select_validate(self):
        @osh.function
        def linear_to_srgb(color: osh.vec3) -> osh.vec3:
            return osh.select(
                color <= osh.vec3(0.0031308),
                color * 12.92,
                1.055 * osh.power(color, osh.vec3(1.0 / 2.4)) - 0.055,
            )

        glsl = osh.compile_function(linear_to_srgb)
        self.assertIn("lessThanEqual(color, vec3(0.0031308))", glsl.source)
        self.assertIn("pow(color, vec3((1.0 / 2.4)))", glsl.source)
        self.assertIn("return mix(", glsl.source)
        wgsl = osh.compile_function(
            linear_to_srgb, target="wgsl", validate=True,
        )
        self.assertIn("(color <= vec3<f32>(0.0031308))", wgsl.source)
        self.assertIn("return select(", wgsl.source)

    def test_typed_locals_and_if_else_validate_for_both_targets(self):
        @osh.function
        def threshold(value: osh.f32, cutoff: osh.f32) -> osh.f32:
            doubled = value * 2.0
            if doubled <= cutoff:
                return doubled
            else:
                return cutoff

        glsl = osh.compile_function(threshold)
        self.assertIn("float doubled = (value * 2.0);", glsl.source)
        self.assertIn("if ((doubled <= cutoff))", glsl.source)
        wgsl = osh.compile_function(threshold, target="wgsl", validate=True)
        self.assertIn("let doubled: f32 = (value * 2.0);", wgsl.source)
        self.assertIn("if ((doubled <= cutoff))", wgsl.source)

    def test_if_rejects_vector_condition(self):
        @osh.function
        def invalid(color: osh.vec3) -> osh.vec3:
            if color <= osh.vec3(1.0):
                return color
            return osh.vec3(0.0)

        with self.assertRaisesRegex(osh.ShaderTypeError, "if condition must be bool"):
            osh.compile_function(invalid)

    def test_vector_select_condition_can_be_a_typed_local(self):
        @osh.function
        def choose(color: osh.vec3) -> osh.vec3:
            low = color <= osh.vec3(0.5)
            return osh.select(low, color, osh.vec3(0.0))

        glsl = osh.compile_function(choose)
        self.assertIn("bvec3 low = lessThanEqual", glsl.source)
        self.assertIn("return mix(vec3(0.0), color, low);", glsl.source)
        osh.compile_function(choose, target="wgsl", validate=True)

    def test_compute_shader_emits_vulkan_glsl_and_reflection(self):
        result = osh.compile(copy_image)
        self.assertEqual(result.target, "glsl")
        self.assertIsNone(result.binary)
        self.assertIn("#version 460", result.source)
        self.assertIn("local_size_x = 8", result.source)
        self.assertIn("readonly image2D source", result.source)
        self.assertIn("writeonly image2D target", result.source)
        self.assertIn(
            "ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);", result.source,
        )
        self.assertIn("imageStore(target, pixel, (color * 0.5));", result.source)
        self.assertEqual(result.reflection.stage, "compute")
        self.assertEqual(
            [(item.name, item.binding) for item in result.reflection.resources],
            [("source", 0), ("target", 1)],
        )

    def test_compute_shader_emits_wgsl_and_portable_reflection(self):
        result = osh.compile(copy_image, target="wgsl")
        self.assertEqual(result.target, "wgsl")
        self.assertIsNone(result.binary)
        self.assertIn("@group(0) @binding(0) var source: texture_storage_2d<rgba16float, read>;", result.source)
        self.assertIn("@group(0) @binding(1) var target_: texture_storage_2d<rgba16float, write>;", result.source)
        self.assertIn("@compute @workgroup_size(8, 4, 1)", result.source)
        self.assertIn("@builtin(global_invocation_id) global_invocation_id: vec3<u32>", result.source)
        self.assertIn(
            "let pixel: vec2<i32> = vec2<i32>(global_invocation_id.xy);",
            result.source,
        )
        self.assertIn(
            "let color: vec4<f32> = textureLoad(source, pixel);", result.source,
        )
        self.assertIn("textureStore(target_, pixel, (color * 0.5));", result.source)
        self.assertEqual(
            [(item.name, item.set, item.binding) for item in result.reflection.resources],
            [("source", 0, 0), ("target", 0, 1)],
        )

    def test_typed_helper_function_emits_wgsl(self):
        @osh.function
        def tint(color: osh.vec3, target: osh.vec3, strength: osh.f32) -> osh.vec3:
            return osh.mix(color, target, strength)

        result = osh.compile_function(tint, target="wgsl")
        self.assertEqual(result.target, "wgsl")
        self.assertIn(
            "fn tint(color: vec3<f32>, target_: vec3<f32>, strength: f32) -> vec3<f32>",
            result.source,
        )
        self.assertIn("return mix(color, target_, strength);", result.source)

    def test_wgsl_select_preserves_ordinaryshade_semantics(self):
        @osh.function
        def choose(condition: osh.boolean, yes: osh.vec3, no: osh.vec3) -> osh.vec3:
            return osh.select(condition, yes, no)

        result = osh.compile_function(choose, target="wgsl")
        self.assertIn("return select(no, yes, condition);", result.source)

    def test_wgsl_storage_format_mapping(self):
        @osh.compute()
        def formats(
            color: osh.storage_image("rgba8", access="read"),
            scalar: osh.storage_image("r32ui", access="write", set=2, binding=5),
        ):
            pass

        result = osh.compile(formats, target="wgsl")
        self.assertIn("texture_storage_2d<rgba8unorm, read>", result.source)
        self.assertIn("@group(2) @binding(5) var scalar: texture_storage_2d<r32uint, write>;", result.source)

    def test_unknown_target_is_rejected(self):
        with self.assertRaisesRegex(osh.ShaderTypeError, "target must be"):
            osh.compile(copy_image, target="metal")

    def test_glsl_storage_image_scalar_types_are_preserved(self):
        @osh.compute()
        def scalar_images(
            unsigned_image: osh.storage_image("r32ui", access="read"),
            signed_image: osh.storage_image("r32i", access="read"),
        ):
            pixel = osh.ivec2(0)
            unsigned_value = unsigned_image.load(pixel).x
            signed_value = signed_image.load(pixel).x
            if unsigned_value == osh.u32(0) and signed_value == osh.i32(0):
                return

        source = osh.compile(scalar_images).source
        self.assertIn("readonly uimage2D unsigned_image", source)
        self.assertIn("readonly iimage2D signed_image", source)
        self.assertIn("uint unsigned_value", source)
        self.assertIn("int signed_value", source)

    def test_glsl_storage_image_descriptor_arrays_and_formatless_output(self):
        @osh.compute()
        def present(
            outputs: osh.storage_image_array(
                "unformatted", 8, access="write", binding=7
            ),
        ):
            index = osh.i32(0)
            outputs[index].store(osh.ivec2(0), osh.vec4(1.0))

        source = osh.compile(present).source
        self.assertIn(
            "layout(set = 0, binding = 7) uniform writeonly image2D outputs[8]",
            source,
        )
        self.assertIn("imageStore(outputs[index]", source)
        with self.assertRaisesRegex(
            osh.ShaderTypeError, "descriptor arrays are not yet available"
        ):
            osh.compile(present, target="wgsl")

    @unittest.skipUnless(shutil.which("naga"), "naga-cli is not installed")
    def test_naga_accepts_generated_wgsl(self):
        result = osh.compile(copy_image, target="wgsl", validate=True)
        self.assertEqual(result.target, "wgsl")

        @osh.function
        def tint(color: osh.vec3, target: osh.vec3, strength: osh.f32) -> osh.vec3:
            return osh.mix(color, target, strength)

        helper = osh.compile_function(tint, target="wgsl", validate=True)
        self.assertEqual(helper.target, "wgsl")

    def test_validation_reports_missing_validator(self):
        with self.assertRaisesRegex(osh.CompilerUnavailableError, "install naga-cli"):
            osh.compile(
                copy_image,
                target="wgsl",
                validate=True,
                wgsl_validator="ordinaryshade-validator-does-not-exist",
            )

    def test_explicit_bindings_are_reflected(self):
        @osh.compute()
        def explicit(
            image: osh.storage_image("rgba32f", binding=4),
        ):
            pass

        result = osh.compile(explicit)
        self.assertEqual(result.reflection.resources[0].binding, 4)

    def test_boolean_conditional_and_break_are_portable(self):
        @osh.function
        def classify(value: osh.i32, enabled: osh.boolean) -> osh.i32:
            result = 0
            for index in range(4):
                if index >= value and enabled:
                    result = 2 if value > 0 or index > 1 else 1
                    break
            return result

        glsl = osh.compile_function(classify).source
        self.assertIn("((index >= value) && enabled)", glsl)
        self.assertIn("((value > 0) || (index > 1)) ? 2 : 1", glsl)
        self.assertIn("break;", glsl)
        wgsl = osh.compile_function(classify, target="wgsl").source
        self.assertIn("((index >= value) && enabled)", wgsl)
        self.assertIn("select(1, 2, ((value > 0) || (index > 1)))", wgsl)
        self.assertIn("break;", wgsl)

    def test_storage_record_supports_a_runtime_structure_array(self):
        @osh.structure
        class Item:
            value: osh.vec4

        @osh.structure
        class Queue:
            count: osh.u32
            capacity: osh.u32
            items: osh.runtime_array(Item)

        @osh.compute()
        def fill(queue: osh.storage_record(Queue, binding=2)):
            index = osh.global_invocation_id.x
            if index >= queue.capacity:
                return
            queue.items[index].value = osh.vec4(1.0)

        glsl = osh.compile(fill).source
        self.assertIn("struct Item", glsl)
        self.assertIn("Item items[];", glsl)
        self.assertIn("queue.items[index].value", glsl)
        wgsl = osh.compile(fill, target="wgsl").source
        self.assertIn("items: array<Item>", wgsl)
        self.assertIn("queue.items[index].value", wgsl)

    def test_glsl_storage_record_does_not_emit_unsized_wrapper_struct(self):
        @osh.structure
        class Item:
            value: osh.f32

        @osh.structure
        class Queue:
            count: osh.u32
            items: osh.runtime_array(Item)

        @osh.compute()
        def consume(queue: osh.storage_record(Queue, binding=0)):
            queue.items[osh.u32(0)].value = osh.f32(queue.count)

        source = osh.compile(consume).source
        self.assertNotIn("struct Queue", source)
        self.assertIn("buffer queue_Block", source)
        self.assertIn("Item items[];", source)

    def test_glsl_reserved_parameter_names_are_escaped(self):
        @osh.function
        def weight(sample: osh.f32) -> osh.f32:
            return sample

        source = osh.compile_function(weight).source
        self.assertIn("float weight(float sample_)", source)
        self.assertIn("return sample_;", source)

    def test_glsl_compute_emits_helper_forward_declarations(self):
        @osh.function
        def inner(value: osh.f32) -> osh.f32:
            return value * 2.0

        @osh.function
        def outer(value: osh.f32) -> osh.f32:
            return inner(value)

        @osh.compute()
        def run(output: osh.storage_buffer(osh.f32, binding=0)):
            output[osh.u32(0)] = outer(1.0)

        source = osh.compile(run, helpers=(outer, inner)).source
        declaration = source.index("float inner(float value);")
        definition = source.index("float inner(float value)\n{")
        self.assertLess(declaration, definition)

    def test_geometry_and_trigonometry_intrinsics_are_portable(self):
        @osh.function
        def orient(a: osh.vec3, b: osh.vec3, angle: osh.f32) -> osh.vec3:
            return osh.cross(a, b) + osh.length(a) * (
                osh.cosine(angle) * a + osh.sine(angle) * b
            )

        glsl = osh.compile_function(orient).source
        self.assertIn("cross(a, b)", glsl)
        self.assertIn("length(a)", glsl)
        self.assertIn("cos(angle)", glsl)
        self.assertIn("sin(angle)", glsl)
        wgsl = osh.compile_function(orient, target="wgsl").source
        self.assertIn("cross(a, b)", wgsl)
        self.assertIn("length(a)", wgsl)

    def test_vulkan_ray_query_resource_and_loop_compile(self):
        @osh.compute(workgroup_size=(64, 1, 1))
        def query_scene(scene: osh.acceleration_structure(binding=0)):
            query = osh.ray_query()
            query.initialize(
                scene, osh.u32(1), osh.u32(1), osh.vec3(0.0), 0.001,
                osh.vec3(0.0, 0.0, 1.0), 1000.0,
            )
            while query.proceed():
                pass
            kind = query.intersection_type(True)
            if kind == osh.u32(1):
                distance = query.intersection_t(True)

        result = osh.compile(query_scene)
        self.assertIn("#extension GL_EXT_ray_query : require", result.source)
        self.assertIn("uniform accelerationStructureEXT scene", result.source)
        self.assertIn("rayQueryEXT query;", result.source)
        self.assertIn("rayQueryInitializeEXT(query, scene", result.source)
        self.assertIn("while (rayQueryProceedEXT(query))", result.source)
        self.assertIn("rayQueryGetIntersectionTEXT(query, true)", result.source)
        self.assertEqual(result.reflection.resources[0].kind, "acceleration_structure")
        with self.assertRaisesRegex(osh.ShaderTypeError, "WGSL does not support"):
            osh.compile(query_scene, target="wgsl")

    def test_declared_subgroup_ballot_and_atomics_compile(self):
        @osh.structure
        class Counters:
            count: osh.u32

        @osh.compute(capabilities=("subgroup_ballot",))
        def compact(counters: osh.storage_record(Counters)):
            belongs = osh.global_invocation_id.x > osh.u32(0)
            ballot = osh.subgroup_ballot(belongs)
            amount = osh.subgroup_ballot_bit_count(ballot)
            base = osh.u32(0)
            if osh.subgroup_elect() and amount > osh.u32(0):
                base = osh.atomic_add(counters.count, amount)
            base = osh.subgroup_broadcast_first(base)
            offset = osh.subgroup_ballot_exclusive_bit_count(ballot)

        result = osh.compile(compact)
        self.assertIn("GL_KHR_shader_subgroup_basic", result.source)
        self.assertIn("GL_KHR_shader_subgroup_ballot", result.source)
        self.assertIn("subgroupBallot(belongs)", result.source)
        self.assertIn("atomicAdd(counters.count, amount)", result.source)
        with self.assertRaisesRegex(osh.ShaderTypeError, "subgroup ballot"):
            osh.compile(compact, target="wgsl")

    def test_compute_scheduling_builtins_and_barrier_are_portable(self):
        @osh.compute(workgroup_size=(8, 4, 1))
        def schedule():
            tile = osh.shared(osh.u32)
            local = osh.local_invocation_id.xy
            lane = osh.local_invocation_index
            group = osh.workgroup_id.xy
            groups = osh.num_workgroups.xy
            osh.workgroup_barrier()

        glsl = osh.compile(schedule).source
        self.assertIn("gl_LocalInvocationID.xy", glsl)
        self.assertIn("shared uint tile;", glsl)
        self.assertIn("gl_LocalInvocationIndex", glsl)
        self.assertIn("gl_WorkGroupID.xy", glsl)
        self.assertIn("gl_NumWorkGroups.xy", glsl)
        self.assertIn("barrier();", glsl)
        wgsl = osh.compile(schedule, target="wgsl").source
        self.assertIn("@builtin(local_invocation_id)", wgsl)
        self.assertIn("var<workgroup> tile: u32;", wgsl)
        self.assertIn("workgroupBarrier();", wgsl)

    def test_reusable_function_hoists_shared_storage(self):
        @osh.function
        def schedule() -> osh.void:
            tile = osh.shared(osh.u32)
            if osh.local_invocation_index == osh.u32(0):
                tile = osh.u32(3)
            osh.workgroup_barrier()

        glsl = osh.compile_function(schedule).source
        self.assertIn("shared uint tile;", glsl)
        self.assertIn("tile = uint(3);", glsl)
        wgsl = osh.compile_function(schedule, target="wgsl").source
        self.assertIn("var<workgroup> tile: u32;", wgsl)
        self.assertIn("tile = u32(3);", wgsl)

    def test_shader_invocation_reordering_is_vulkan_specific(self):
        @osh.compute(capabilities=("shader_reorder",))
        def reorder():
            osh.reorder_thread(osh.u32(7), osh.u32(31))

        glsl = osh.compile(reorder).source
        self.assertIn("GL_NV_shader_invocation_reorder", glsl)
        self.assertIn("reorderThreadNV(uint(7), uint(31));", glsl)
        with self.assertRaisesRegex(osh.ShaderTypeError, "reordering"):
            osh.compile(reorder, target="wgsl")

        @osh.function
        def reorder_fragment(hint: osh.u32) -> osh.void:
            osh.reorder_thread(hint, osh.u32(31))

        fragment = osh.compile_function(
            reorder_fragment, capabilities=("shader_reorder",)
        ).source
        self.assertIn("reorderThreadNV(hint, uint(31));", fragment)

    def test_external_abi_and_inout_parameters_compose_without_raw_source(self):
        PathState = osh.opaque_type("PathState")

        @osh.external
        def integrate(path: osh.inout(PathState), distance: osh.f32) -> osh.void:
            pass

        @osh.compute()
        def run(states: osh.storage_buffer(PathState)):
            integrate(states[osh.global_invocation_id.x], 10.0)

        source = osh.compile(run, externals=(integrate,)).source
        self.assertIn("void integrate(inout PathState path, float distance);", source)
        self.assertIn("integrate(states[gl_GlobalInvocationID.x], 10.0);", source)
        with self.assertRaisesRegex(osh.ShaderTypeError, "external function"):
            osh.compile(run, target="wgsl", externals=(integrate,))

        @osh.function
        def embedded_trace(origin: osh.vec3) -> osh.f32:
            query = osh.ray_query()
            query.initialize(
                scene_tlas, osh.u32(1), osh.u32(1), origin, 0.001,
                osh.vec3(0.0, 0.0, 1.0), 1000.0,
            )
            while query.proceed():
                pass
            return query.intersection_t(True)

        fragment = osh.compile_function(
            embedded_trace,
            external_values={
                "scene_tlas": osh.opaque_type("accelerationStructureEXT")
            },
        ).source
        self.assertIn("rayQueryInitializeEXT(query, scene_tlas", fragment)
        self.assertNotIn("uniform accelerationStructureEXT", fragment)

    def test_fixed_arrays_in_structures_are_portable_and_indexable(self):
        @osh.structure
        class Stack:
            values: osh.array(osh.f32, 16)

        @osh.compute()
        def update(stacks: osh.storage_buffer(Stack)):
            stacks[osh.global_invocation_id.x].values[osh.u32(3)] = 1.5

        glsl = osh.compile(update).source
        self.assertIn("float values[16];", glsl)
        self.assertIn(".values[uint(3)] = 1.5", glsl)
        wgsl = osh.compile(update, target="wgsl").source
        self.assertIn("values: array<f32, 16>", wgsl)

        @osh.compute()
        def undeclared(image: osh.storage_image("rgba16f")):
            elected = osh.subgroup_elect()

        with self.assertRaisesRegex(osh.ShaderTypeError, "require capabilities"):
            osh.compile(undeclared)

    def test_helpers_can_share_resources_and_return_structures_or_void(self):
        @osh.structure
        class Record:
            value: osh.vec2
            valid: osh.boolean

        @osh.function
        def load_record(index: osh.u32) -> Record:
            return Record(osh.unpack_half2x16(words[index]), True)

        @osh.function
        def store_record(index: osh.u32, record: Record) -> osh.void:
            words[index] = osh.pack_half2x16(record.value)

        @osh.compute()
        def copy_record(words: osh.storage_buffer(osh.u32)):
            record = load_record(osh.u32(0))
            if record.valid:
                store_record(osh.u32(1), record)
            osh.memory_barrier_buffer()

        source = osh.compile(
            copy_record, helpers=(load_record, store_record)
        ).source
        self.assertIn("struct Record", source)
        self.assertIn("Record load_record(uint index)", source)
        self.assertIn("void store_record(uint index, Record record)", source)
        self.assertIn("unpackHalf2x16(words[index])", source)
        self.assertIn("words[index] = packHalf2x16(record.value)", source)
        self.assertIn("memoryBarrierBuffer()", source)
        wgsl = osh.compile(
            copy_record, target="wgsl", helpers=(load_record, store_record)
        ).source
        self.assertIn("fn store_record(index: u32, record: Record) {", wgsl)
        self.assertIn("unpack2x16float(words[index])", wgsl)

    def test_entrypoint_capabilities_are_available_to_helpers(self):
        @osh.function
        def compact_index() -> osh.u32:
            lanes = osh.subgroup_ballot(True)
            return osh.subgroup_ballot_exclusive_bit_count(lanes)

        @osh.compute(capabilities=("subgroup_ballot",))
        def compact(output: osh.storage_buffer(osh.u32)):
            output[osh.global_invocation_id.x] = compact_index()

        source = osh.compile(compact, helpers=(compact_index,)).source
        self.assertIn("subgroupBallotExclusiveBitCount", source)

    def test_unsupported_statement_has_shader_diagnostic(self):
        @osh.compute()
        def invalid(image: osh.storage_image("rgba16f")):
            try:
                image.store(osh.global_invocation_id.xy, 0)
            except RuntimeError:
                pass

        with self.assertRaisesRegex(osh.ShaderSyntaxError, "unsupported shader statement"):
            osh.compile(invalid)

    def test_resource_validation(self):
        with self.assertRaises(osh.ShaderTypeError):
            osh.storage_image("not-a-format")
        with self.assertRaises(osh.ShaderTypeError):
            osh.storage_image("rgba16f", access="sometimes")

    def test_scalar_bitcasts_are_portable(self):
        @osh.function
        def round_trip(value: osh.f32) -> osh.f32:
            return osh.uint_bits_to_float(osh.float_bits_to_uint(value))

        glsl = osh.compile_function(round_trip).source
        self.assertIn("uintBitsToFloat(floatBitsToUint(value))", glsl)
        wgsl = osh.compile_function(round_trip, target="wgsl").source
        self.assertIn("bitcast<f32>(bitcast<u32>(value))", wgsl)

    def test_refract_is_portable(self):
        @osh.function
        def transmit(incoming: osh.vec3, normal: osh.vec3) -> osh.vec3:
            return osh.refract(incoming, normal, 0.75)

        glsl = osh.compile_function(transmit).source
        self.assertIn("refract(incoming, normal, 0.75)", glsl)
        wgsl = osh.compile_function(transmit, target="wgsl").source
        self.assertIn("refract(incoming, normal, 0.75)", wgsl)

    def test_bitfield_reverse_is_portable(self):
        @osh.function
        def reverse(value: osh.u32) -> osh.u32:
            return osh.bitfield_reverse(value)

        glsl = osh.compile_function(reverse).source
        self.assertIn("bitfieldReverse(value)", glsl)
        wgsl = osh.compile_function(reverse, target="wgsl").source
        self.assertIn("reverseBits(value)", wgsl)

    def test_angular_and_fraction_intrinsics_are_portable(self):
        @osh.function
        def environment_uv(direction: osh.vec3) -> osh.vec2:
            return osh.vec2(
                osh.fraction(osh.arctangent2(direction.z, direction.x)),
                osh.arccosine(osh.clamp(direction.y, -1.0, 1.0)),
            )

        glsl = osh.compile_function(environment_uv).source
        self.assertIn("fract(atan(direction.z, direction.x))", glsl)
        self.assertIn("acos(clamp(direction.y", glsl)
        wgsl = osh.compile_function(environment_uv, target="wgsl").source
        self.assertIn("fract(atan2(direction.z, direction.x))", wgsl)

    def test_matrix_vector_product_infers_vector_result(self):
        @osh.function
        def transform(matrix: osh.mat4, point: osh.vec3) -> osh.vec3:
            return (matrix * osh.vec4(point, 1.0)).xyz

        glsl = osh.compile_function(transform).source
        self.assertIn("(matrix * vec4(point, 1.0)).xyz", glsl)
        wgsl = osh.compile_function(transform, target="wgsl").source
        self.assertIn("(matrix * vec4<f32>(point, 1.0)).xyz", wgsl)

    def test_logarithm_and_ceiling_are_portable(self):
        @osh.function
        def optical_depth(alpha: osh.f32) -> osh.f32:
            return -osh.logarithm(1.0 - alpha) + osh.ceiling(alpha)

        glsl = osh.compile_function(optical_depth).source
        self.assertIn("log((1.0 - alpha))", glsl)
        self.assertIn("ceil(alpha)", glsl)
        wgsl = osh.compile_function(optical_depth, target="wgsl").source
        self.assertIn("log((1.0 - alpha))", wgsl)

    def test_vulkan_sampled_texture_3d_array(self):
        @osh.compute()
        def sample_volume(
            volumes: osh.sampled_texture_3d_array(16, binding=3),
            output: osh.storage_buffer(osh.f32, binding=4),
        ):
            output[osh.global_invocation_id.x] = volumes.sample(
                osh.u32(2), osh.vec3(0.5)
            )

        result = osh.compile(sample_volume)
        self.assertIn("uniform sampler3D volumes[16]", result.source)
        self.assertIn("texture(volumes[uint(2)], vec3(0.5)).r", result.source)
        self.assertEqual(result.reflection.resources[0].kind, "sampled_texture_3d_array")
        with self.assertRaisesRegex(osh.ShaderTypeError, "combined-sampler"):
            osh.compile(sample_volume, target="wgsl")

    def test_vulkan_sampled_texture_2d_array(self):
        @osh.compute()
        def sample_texture(
            textures: osh.sampled_texture_2d_array(128, binding=3),
            output: osh.storage_buffer(osh.vec4, binding=4),
        ):
            index = osh.global_invocation_id.x
            size = textures.size(index, 0)
            levels = textures.levels(index)
            output[index] = textures.sample_lod(
                index, osh.vec2(size) * 0.0, osh.f32(levels - 1)
            )

        result = osh.compile(sample_texture)
        self.assertIn("#extension GL_EXT_nonuniform_qualifier : require", result.source)
        self.assertIn("uniform sampler2D textures[128]", result.source)
        self.assertIn("textureSize(textures[nonuniformEXT(index)], 0)", result.source)
        self.assertIn("textureQueryLevels(textures[nonuniformEXT(index)])", result.source)
        self.assertIn("textureLod(textures[nonuniformEXT(index)]", result.source)
        self.assertEqual(result.reflection.resources[0].kind, "sampled_texture_2d_array")
        with self.assertRaisesRegex(osh.ShaderTypeError, "combined-sampler"):
            osh.compile(sample_texture, target="wgsl")

    def test_function_local_fixed_arrays_are_portable(self):
        @osh.function
        def accumulate(index: osh.u32, value: osh.f32) -> osh.f32:
            entries = osh.local_array(osh.f32, 16)
            entries[index] = value
            return entries[index]

        glsl = osh.compile_function(accumulate).source
        self.assertIn("float entries[16];", glsl)
        self.assertIn("entries[index] = value;", glsl)
        wgsl = osh.compile_function(accumulate, target="wgsl").source
        self.assertIn("var entries: array<f32, 16>;", wgsl)
        self.assertIn("entries[index] = value;", wgsl)

    def test_workgroup_validation(self):
        with self.assertRaises(osh.ShaderTypeError):
            osh.compute(workgroup_size=(8, 0, 1))


if __name__ == "__main__":
    unittest.main()
def test_compiled_shader_has_stable_identity_and_source_map():
    @osh.compute(workgroup_size=(1, 1, 1))
    def identity(output: osh.storage_image("rgba16f", access="write", set=0, binding=0)):
        output.store(osh.global_invocation_id.xy, osh.vec4(1.0))

    first = osh.compile(identity)
    second = osh.compile(identity)
    wgsl = osh.compile(identity, target="wgsl")
    assert first.cache_key == second.cache_key
    assert first.cache_key != wgsl.cache_key
    assert len(first.cache_key) == 64
    assert first.source_map
    assert first.source_map[0].source.path.endswith("test_compiler.py")
