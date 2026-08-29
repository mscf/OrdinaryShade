"""Python AST to typed shader IR lowering."""
from . import _core
globals().update({name: value for name, value in vars(_core).items() if not name.startswith("_")})
