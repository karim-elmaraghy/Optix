"""OptiLoop: find the fastest correct Llama-3.2-1B inference config.

Subtask A (this package's benchmark half): workload, profiles, benchmark,
run. Quality validation, the Meta agent, and safe apply live in sibling
modules (``validator``, ``agent``, ``safe_apply``) owned by Subtask C.
No GPU-library imports at package level.
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
