"""
Utility functions for LLM Scaling Analysis.

This package contains general utility functions for colored output,
environment file loading, and mathematical expression conversion.
"""

import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from training_models.utils.colors import (
    print_red,
    print_blue,
    print_green,
    print_yellow,
    print_purple,
    print_cyan,
    print_bold,
    print_underline,
    colored_text,
    print_colored,
    print_header,
    print_section,
    print_success,
    print_error,
    print_warning,
    print_info,
)
from training_models.utils.env import load_env_file

from training_models.utils.latex import (
    convert_expression_to_latex,
    normalize_fraction_by_leading_coefficient,
    break_latex_equation,
)

__all__ = [
    "print_red",
    "print_blue",
    "print_green",
    "print_yellow",
    "print_purple",
    "print_cyan",
    "print_bold",
    "print_underline",
    "colored_text",
    "print_colored",
    "print_header",
    "print_section",
    "print_success",
    "print_error",
    "print_warning",
    "print_info",
    "convert_expression_to_latex",
    "load_env_file",
    "normalize_fraction_by_leading_coefficient",
    "break_latex_equation",
]
