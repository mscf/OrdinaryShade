import unittest
import ordinaryshade as osh


@osh.compute()
def small_storage_outputs(
    motion: osh.storage_image('rg16f', access='write', binding=0),
    reactive: osh.storage_image('r8', access='write', binding=1),
):
    pixel = osh.ivec2(osh.global_invocation_id.xy)
    motion.store(pixel, osh.vec4(1.0, 2.0, 0.0, 0.0))
    reactive.store(pixel, osh.vec4(0.5))


class AdditionalStorageFormatTests(unittest.TestCase):
    def test_glsl_preserves_formats(self):
        source = osh.compile(small_storage_outputs).source
        self.assertIn('rg16f)', source)
        self.assertIn('r8)', source)

    def test_wgsl_reports_unsupported_format(self):
        with self.assertRaisesRegex(osh.ShaderTypeError, 'storage format'):
            osh.compile(small_storage_outputs, target='wgsl')
