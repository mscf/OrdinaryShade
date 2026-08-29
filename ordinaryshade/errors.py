"""Compiler diagnostics exposed by Ordinary Shade."""


class ShaderError(Exception):
    """Base class for Ordinary Shade errors."""


class ShaderSyntaxError(ShaderError):
    """The Python shader uses syntax outside the supported language."""


class ShaderTypeError(ShaderError):
    """The shader contains an invalid type or resource operation."""


class CompilerUnavailableError(ShaderError):
    """An optional external compiler could not be located."""


class ShaderCompilationError(ShaderError):
    """A backend compiler rejected generated shader source."""

