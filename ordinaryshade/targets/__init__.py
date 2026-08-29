"""Ordinary Shade compiler targets."""

from .glsl import emit_glsl
from .wgsl import emit_wgsl

__all__ = ["emit_glsl", "emit_wgsl"]
