from pathlib import Path
import unittest

import ordinaryshade as osh


class PackageOrganizationTests(unittest.TestCase):
    def test_root_contains_only_package_entrypoint(self):
        root = Path(osh.__file__).parent
        self.assertEqual(
            {path.name for path in root.glob("*.py")},
            {"__init__.py"},
        )

    def test_semantic_namespaces_preserve_public_api(self):
        from ordinaryshade.compiler.compiled_shader import CompiledShader
        from ordinaryshade.entrypoints.compute_shader import ComputeShader
        from ordinaryshade.reflection import ShaderReflection as PublicShaderReflection
        from ordinaryshade.reflection.shader_reflection import ShaderReflection
        from ordinaryshade.types.shader_type import ShaderType

        self.assertIs(CompiledShader, osh.CompiledShader)
        self.assertIs(ComputeShader, osh.ComputeShader)
        self.assertIs(ShaderReflection, PublicShaderReflection)
        self.assertIs(ShaderType, osh.ShaderType)

    def test_codegen_targets_are_explicit(self):
        from ordinaryshade.targets import emit_glsl, emit_wgsl

        self.assertTrue(callable(emit_glsl))
        self.assertTrue(callable(emit_wgsl))


if __name__ == "__main__":
    unittest.main()
