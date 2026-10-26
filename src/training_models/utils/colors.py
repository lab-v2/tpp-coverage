#!/usr/bin/env python3
"""
Color and printing utilities for LLM Scaling Analysis.
"""

from typing import Any

# ANSI color code constants
RED = "\033[91m"
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
PURPLE = "\033[95m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"
UNDERLINE = "\033[4m"


def print_red(text: Any):
    """Print text in red color."""
    print(f"\033[91m{text}\033[0m")


def print_blue(text: Any):
    """Print text in blue color."""
    print(f"\033[94m{text}\033[0m")


def print_green(text: Any):
    """Print text in green color."""
    print(f"\033[92m{text}\033[0m")


def print_yellow(text: Any):
    """Print text in yellow color."""
    print(f"\033[93m{text}\033[0m")


def print_purple(text: Any):
    """Print text in purple/magenta color."""
    print(f"\033[95m{text}\033[0m")


def print_cyan(text: Any):
    """Print text in cyan color."""
    print(f"\033[96m{text}\033[0m")


def print_bold(text: Any):
    """Print text in bold."""
    print(f"\033[1m{text}\033[0m")


def print_underline(text: Any):
    """Print text with underline."""
    print(f"\033[4m{text}\033[0m")


def colored_text(text: Any, color: str) -> str:
    """
    Return text with color formatting (without printing).

    Args:
        text: Text to colorize
        color: Color name ('red', 'blue', 'green', 'yellow', 'purple', 'cyan', 'bold', 'underline')

    Returns:
        Formatted string with color codes
    """
    color_codes = {
        "red": "\033[91m",
        "blue": "\033[94m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "purple": "\033[95m",
        "cyan": "\033[96m",
        "bold": "\033[1m",
        "underline": "\033[4m",
    }

    reset_code = "\033[0m"
    color_code = color_codes.get(color.lower(), "")

    return f"{color_code}{text}{reset_code}" if color_code else str(text)


def print_colored(text: Any, color: str):
    """
    Print text in the specified color.

    Args:
        text: Text to print
        color: Color name ('red', 'blue', 'green', 'yellow', 'purple', 'cyan', 'bold', 'underline')
    """
    print(colored_text(text, color))


def print_header(title: str, color: str = "blue", width: int = 65):
    """
    Print a formatted header with color and borders.

    Args:
        title: Header title
        color: Color for the header
        width: Width of the header border
    """
    border = "=" * width
    print_colored(border, color)
    print_colored(title.center(width), color)
    print_colored(border, color)


def print_section(title: str, color: str = "yellow"):
    """
    Print a section separator with color.

    Args:
        title: Section title
        color: Color for the section title
    """
    print()
    print_colored(f"--- {title} ---", color)


def print_success(message: Any):
    """Print success message in green with checkmark."""
    print_green(f"✓ {message}")


def print_error(message: Any):
    """Print error message in red with X mark."""
    print_red(f"✗ {message}")


def print_warning(message: Any):
    """Print warning message in yellow with warning symbol."""
    print_yellow(f"⚠ {message}")


def print_info(message: Any):
    """Print info message in blue with info symbol."""
    print_blue(f"ℹ {message}")
