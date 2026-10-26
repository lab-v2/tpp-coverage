#!/usr/bin/env python3
"""
Utility functions for LLM Scaling Analysis

This module contains general utility functions for colored output, environment file loading,
and mathematical expression conversion.
"""

from functools import reduce
from math import gcd
from pathlib import Path
from typing import Any
import sympy as sp


def normalize_fraction_by_leading_coefficient(expr: sp.Expr) -> sp.Expr:
    """
    Normalize a fraction by factoring out the leading coefficient.
    
    GUARANTEES: After normalization, the polynomial part has leading coefficient = 1.
    
    Process:
    1. Simplify the whole expression
    2. Factor out the leading coefficient as: scalar_factor × polynomial
    3. Simplify both parts separately
    
    Example: (2x^2 + 4y) / (3xy) -> (2/(3xy)) × (x^2 + 2y)
    where the polynomial (x^2 + 2y) has leading coefficient 1.
    
    Args:
        expr: SymPy expression (should be a rational function)
    
    Returns:
        Normalized SymPy expression in factored form: scalar × polynomial(leading_coeff=1)
    """
    # STEP 1: Simplify the whole expression first
    simplified_expr = sp.simplify(expr)
    simplified_expr = sp.cancel(simplified_expr)
    
    # Extract numerator and denominator
    numer, denom = sp.fraction(simplified_expr)
    
    # Get all symbols in the numerator
    symbols = list(numer.free_symbols)
    
    if not symbols:
        # If no symbols, just return simplified form
        return simplified_expr
    
    # Expand numerator to see all terms clearly
    numer_expanded = sp.expand(numer)
    
    # Get all terms in the numerator
    terms = sp.Add.make_args(numer_expanded)
    
    if not terms:
        return simplified_expr
    
    # Find the term with the highest total degree
    max_degree = -1
    max_term = None
    _max_term_idx = -1
    
    for idx, term in enumerate(terms):
        # Calculate total degree (sum of degrees of all variables)
        total_degree = sum(sp.degree(term, gen) for gen in symbols)
        if total_degree > max_degree:
            max_degree = total_degree
            max_term = term
            _max_term_idx = idx
    
    if max_term is None:
        return simplified_expr
    
    # Extract the coefficient from the highest degree term
    leading_coeff, leading_term_without_coeff = max_term.as_coeff_Mul()
    
    # Only normalize if coefficient is not 1
    if leading_coeff == 1:
        return simplified_expr
    
    # STEP 2: Factor out the leading coefficient
    # Result format: (leading_coeff / denom) × (normalized_polynomial)
    # where normalized_polynomial has leading coefficient = 1
    
    # Convert to rational if needed to avoid floating point issues
    if leading_coeff.is_Integer:
        rational_coeff = sp.Rational(int(leading_coeff), 1)
    elif leading_coeff.is_Rational:
        rational_coeff = leading_coeff
    elif leading_coeff.is_Float:
        rational_coeff = sp.nsimplify(leading_coeff)
    else:
        rational_coeff = leading_coeff
    
    # Divide numerator by leading coefficient to get polynomial with leading coeff = 1
    normalized_poly = numer / rational_coeff
    
    # The scalar factor is leading_coeff / denom
    scalar_factor = rational_coeff / denom
    
    # STEP 3: Simplify the scalar factor and normalized polynomial
    normalized_poly_expanded = sp.expand(normalized_poly)
    scalar_factor_simplified = sp.simplify(scalar_factor)
    
    # Return: scalar_factor × normalized_polynomial
    # This represents (leading_coeff / denom) × (polynomial with leading coeff = 1)
    # The polynomial part has leading coefficient = 1 by construction
    result = scalar_factor_simplified * normalized_poly_expanded
    return result


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
        'red': '\033[91m',
        'blue': '\033[94m', 
        'green': '\033[92m',
        'yellow': '\033[93m',
        'purple': '\033[95m',
        'cyan': '\033[96m',
        'bold': '\033[1m',
        'underline': '\033[4m'
    }
    
    reset_code = '\033[0m'
    color_code = color_codes.get(color.lower(), '')
    
    return f"{color_code}{text}{reset_code}" if color_code else str(text)


def print_colored(text: Any, color: str):
    """
    Print text in the specified color.
    
    Args:
        text: Text to print
        color: Color name ('red', 'blue', 'green', 'yellow', 'purple', 'cyan', 'bold', 'underline')
    """
    print(colored_text(text, color))


def print_header(title: str, color: str = 'blue', width: int = 65):
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


def print_section(title: str, color: str = 'yellow'):
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


def _extract_fraction_parts(latex_str: str) -> tuple[str, str] | None:
    """
    Extract numerator and denominator from a LaTeX fraction, handling nested braces.
    
    Args:
        latex_str: LaTeX string starting with \\frac
        
    Returns:
        Tuple of (numerator, denominator) or None if not a valid fraction
    """
    if not latex_str.startswith('\\frac{'):
        return None
    
    # Start after '\frac{'
    pos = 6
    depth = 1
    numerator_chars = []
    
    # Extract numerator by counting braces
    while pos < len(latex_str) and depth > 0:
        char = latex_str[pos]
        if char == '{':
            depth += 1
            numerator_chars.append(char)
        elif char == '}':
            depth -= 1
            if depth > 0:
                numerator_chars.append(char)
        else:
            numerator_chars.append(char)
        pos += 1
    
    if depth != 0 or pos >= len(latex_str) or latex_str[pos] != '{':
        return None
    
    numerator = ''.join(numerator_chars)
    
    # Extract denominator
    pos += 1  # Skip opening '{'
    depth = 1
    denominator_chars = []
    
    while pos < len(latex_str) and depth > 0:
        char = latex_str[pos]
        if char == '{':
            depth += 1
            denominator_chars.append(char)
        elif char == '}':
            depth -= 1
            if depth > 0:
                denominator_chars.append(char)
        else:
            denominator_chars.append(char)
        pos += 1
    
    if depth != 0:
        return None
    
    denominator = ''.join(denominator_chars)
    
    return numerator, denominator


def _break_fraction_numerator(latex_str: str, max_width: int) -> str:
    """
    Break a fraction by splitting its numerator into multiple lines.
    
    For a fraction like: \\frac{term1 + term2 + term3}{denominator}
    Creates: \\frac{\\begin{aligned}term1 \\\\ + term2 \\\\ + term3\\end{aligned}}{denominator}
    
    Args:
        latex_str: LaTeX fraction string
        max_width: Maximum width per line
        
    Returns:
        LaTeX fraction with multi-line numerator if needed
    """
    # Extract numerator and denominator properly handling nested braces
    parts = _extract_fraction_parts(latex_str)
    
    if parts is None:
        return latex_str
    
    numerator, denominator = parts
    
    # Split numerator at + and - operators, preserving parentheses
    num_terms = _split_at_operators(numerator)
    
    if len(num_terms) <= 1:
        # Can't split, return as-is
        return latex_str
    
    # Build multi-line numerator using aligned environment
    aligned_lines = []
    for i, term in enumerate(num_terms):
        term = term.strip()
        if i == 0:
            # First line: just the term
            aligned_lines.append(term)
        else:
            # Subsequent lines: aligned with + or - prefix
            if term.startswith('-'):
                aligned_lines.append(term)
            elif term.startswith('+'):
                aligned_lines.append(term)
            else:
                aligned_lines.append('+' + term)
    
    # Join with line breaks - use & for alignment in aligned environment
    numerator_multiline = ' \\\\\\\\ '.join(aligned_lines)
    
    # Wrap in aligned environment
    numerator_aligned = f'\\begin{{aligned}}{numerator_multiline}\\end{{aligned}}'
    
    # Reconstruct the fraction
    result = f'\\frac{{{numerator_aligned}}}{{{denominator}}}'
    
    return result


def break_latex_equation(latex_str: str, max_width: int = 120) -> str:
    """
    Break a long LaTeX equation into multiple lines using aligned environment.
    
    This function intelligently splits LaTeX equations at addition/subtraction operators
    and formats them with proper alignment for readability.
    
    Args:
        latex_str: LaTeX equation string (without surrounding $ signs)
        max_width: Approximate maximum character width per line
        
    Returns:
        Formatted LaTeX string with aligned environment if the equation is long
    """
    # If equation is short enough, return as-is
    if len(latex_str) <= max_width:
        return latex_str
    
    # Check if it's a fraction (starts with \frac)
    is_fraction = latex_str.strip().startswith('\\frac')
    
    if is_fraction:
        # For fractions, break the numerator into multiple lines
        return _break_fraction_numerator(latex_str, max_width)
    else:
        # For regular equations, split at + and - operators
        return _break_regular_equation(latex_str, max_width)


def _break_regular_equation(latex_str: str, max_width: int) -> str:
    """Break a regular (non-fraction) equation at addition/subtraction operators."""
    terms = _split_at_operators(latex_str)
    
    if len(terms) <= 1:
        return latex_str
    
    # Build aligned environment
    lines = ["\\begin{aligned}"]
    
    for i, term in enumerate(terms):
        term = term.strip()
        if i == 0:
            lines.append(f"    {term}")
        else:
            if not term.startswith(('+', '-')):
                term = '+ ' + term
            lines.append(f"    &{term}")
    
    lines.append("\\end{aligned}")
    
    return "\n".join(lines)


def _split_at_operators(expr: str) -> list[str]:
    """
    Split a LaTeX expression at top-level + and - operators.
    
    This function respects nesting (braces and parentheses) and only splits
    at operators that are not inside nested structures.
    """
    terms = []
    current_term = []
    depth = 0  # Track nesting depth for braces
    paren_depth = 0  # Track nesting depth for parentheses
    i = 0
    
    while i < len(expr):
        char = expr[i]
        
        # Track brace depth
        if char == '{':
            depth += 1
            current_term.append(char)
        elif char == '}':
            depth -= 1
            current_term.append(char)
        # Track parenthesis depth
        elif char == '(':
            paren_depth += 1
            current_term.append(char)
        elif char == ')':
            paren_depth -= 1
            current_term.append(char)
        # Split at top-level + or - operators
        elif char in ['+', '-'] and depth == 0 and paren_depth == 0:
            # Make sure this isn't part of a number (e.g., 1e-10) or \cdot command
            if i > 0 and expr[i-1] not in ['e', 'E', '\\']:
                # Save current term
                if current_term:
                    terms.append(''.join(current_term).strip())
                # Start new term with the operator
                current_term = [char, ' ']
                i += 1
                continue
            else:
                current_term.append(char)
        else:
            current_term.append(char)
        
        i += 1
    
    # Add the last term
    if current_term:
        terms.append(''.join(current_term).strip())
    
    return terms


def _apply_simplification_pipeline(expr: sp.Expr) -> sp.Expr:
    """
    Apply comprehensive simplification pipeline to a SymPy expression.
    
    Args:
        expr: SymPy expression to simplify
    
    Returns:
        Simplified SymPy expression
    """
    simplified = sp.simplify(expr)
    simplified = sp.cancel(simplified)
    simplified = sp.powsimp(simplified, combine='all', force=True)
    
    try:
        simplified = sp.radsimp(simplified)
    except (ValueError, TypeError, AttributeError):
        # Skip if radsimp fails (can happen with certain expressions)
        pass
    
    try:
        simplified = sp.together(simplified)
    except (ValueError, TypeError, AttributeError):
        # Skip if together fails (can happen with certain expressions)
        pass
    
    # More aggressive rational simplification
    simplified = sp.nsimplify(simplified, rational=True, tolerance=1e-10)
    simplified = sp.factor_terms(simplified)
    
    try:
        simplified = sp.factor(simplified)
    except (ValueError, TypeError, AttributeError):
        # If factoring fails, continue with current form
        pass
    
    simplified = sp.gcd_terms(simplified)
    
    # Multiple rounds of simplification and cancellation for stubborn expressions
    for _ in range(2):
        simplified = sp.simplify(simplified)
        simplified = sp.cancel(simplified)
        
        # Try to extract common numeric factors from fractions
        try:
            numer, denom = sp.fraction(simplified)
            
            # Get all terms
            numer_terms = sp.Add.make_args(sp.expand(numer))
            denom_terms = sp.Add.make_args(sp.expand(denom))
            
            # Extract integer coefficients
            numer_int_coeffs = []
            for term in numer_terms:
                coeff = term.as_coeff_Mul()[0]
                if coeff.is_Number and coeff != 0:
                    try:
                        numer_int_coeffs.append(int(abs(float(coeff))))
                    except (ValueError, OverflowError):
                        # Skip if coefficient can't be converted to int
                        pass
            
            denom_int_coeffs = []
            for term in denom_terms:
                coeff = term.as_coeff_Mul()[0]
                if coeff.is_Number and coeff != 0:
                    try:
                        denom_int_coeffs.append(int(abs(float(coeff))))
                    except (ValueError, OverflowError):
                        # Skip if coefficient can't be converted to int
                        pass
            
            if numer_int_coeffs and denom_int_coeffs:
                numer_gcd = reduce(gcd, numer_int_coeffs)
                denom_gcd = reduce(gcd, denom_int_coeffs)
                common_gcd = gcd(numer_gcd, denom_gcd)
                
                if common_gcd > 1:
                    simplified = (numer / common_gcd) / (denom / common_gcd)
                    simplified = sp.cancel(simplified)
        except (ValueError, TypeError, ZeroDivisionError):
            # Skip GCD simplification if it fails
            pass
    
    return simplified


def _simplify_large_fraction_coefficients(expr: sp.Expr) -> sp.Expr:
    """
    Simplify large numeric coefficients in fractions by extracting common factors.
    
    Args:
        expr: SymPy expression (typically a fraction)
    
    Returns:
        Simplified expression with reduced numeric coefficients
    """
    try:
        numer, denom = sp.fraction(expr)
        
        # Expand to see all terms
        numer_expanded = sp.expand(numer)
        denom_expanded = sp.expand(denom)
        
        # Get all numeric coefficients from both numerator and denominator
        def get_numeric_coeffs(expr_part):
            coeffs = []
            for term in sp.Add.make_args(expr_part):
                coeff = term.as_coeff_Mul()[0]
                if coeff.is_Number and coeff != 0:
                    coeffs.append(abs(float(coeff)))
            return coeffs
        
        numer_coeffs = get_numeric_coeffs(numer_expanded)
        denom_coeffs = get_numeric_coeffs(denom_expanded)
        
        if numer_coeffs and denom_coeffs:
            # Convert to integers if possible
            int_numer_coeffs = [int(c) for c in numer_coeffs if c == int(c)]
            int_denom_coeffs = [int(c) for c in denom_coeffs if c == int(c)]
            
            if int_numer_coeffs and int_denom_coeffs:
                # Find GCD of all coefficients
                numer_gcd = reduce(gcd, int_numer_coeffs)
                denom_gcd = reduce(gcd, int_denom_coeffs)
                common_factor = gcd(numer_gcd, denom_gcd)
                
                if common_factor > 1:
                    # Divide both by common factor
                    simplified = (numer / common_factor) / (denom / common_factor)
                    return sp.cancel(simplified)
        
        return expr
    except (ValueError, TypeError, ZeroDivisionError):
        # If simplification fails, return original expression
        return expr


def _convert_large_numbers_to_float(expr: sp.Expr) -> sp.Expr:
    """
    Convert large numbers to Float objects for scientific notation.
    
    Args:
        expr: SymPy expression
    
    Returns:
        Expression with large numbers converted to Float objects
    """
    def replace_large_numbers(x: sp.Expr) -> sp.Expr:
        if x.is_Number and not x.is_Symbol:
            abs_val = abs(float(x))
            if abs_val >= 1000:
                return sp.Float(x, 3)
        return x
    
    return expr.replace(
        lambda x: x.is_Number and not x.is_Symbol and abs(float(x)) >= 1000, 
        replace_large_numbers
    )


def _check_for_invalid_terms(expr: sp.Expr, context: str) -> tuple[bool, str]:
    """
    Check if expression contains infinity or NaN terms.
    
    Args:
        expr: SymPy expression to check
        context: Context description for error message
    
    Returns:
        Tuple of (has_invalid, error_latex)
    """
    if expr.has(sp.oo, sp.zoo, -sp.oo, sp.nan):
        error_latex = f"\\text{{ERROR: {context} contains infinity or NaN terms}}"
        return True, error_latex
    return False, ""


def convert_expression_to_latex(expression: str) -> tuple[str, str]:
    """
    Convert gplearn mathematical expression to LaTeX format using SymPy.
    
    Uses SymPy's sympify to parse expressions and latex to convert to LaTeX format.
    
    Args:
        expression: String representation of gplearn mathematical expression
    
    Returns:
        Tuple of (original_expression, simplified_expression) in LaTeX format
    """
    expr = expression.replace('X0', 'n').replace('X1', 'm').replace('X2', 'l').replace('X3', 'b')
    
    n, m, l, b = sp.symbols('n m l b')
    
    namespace = {
        'n': n, 'm': m, 'l': l, 'b': b,
        'add': sp.Add,
        'sub': lambda a, b: sp.Add(a, -b),
        'mul': sp.Mul,
        'div': lambda a, b: sp.Mul(a, sp.Pow(b, -1)),
    }
   
    sympy_expr = sp.sympify(expr, locals=namespace)
    
    has_invalid, error_latex = _check_for_invalid_terms(expr=sympy_expr, context="Expression")
    if has_invalid:
        return f"${error_latex}$", f"${error_latex}$"
    
    simplified = _apply_simplification_pipeline(expr=sympy_expr)
    
    has_invalid, error_latex = _check_for_invalid_terms(expr=simplified, context="Simplification")
    if has_invalid:
        return f"${error_latex}$", f"${error_latex}$"
    
    try:
        simplified = normalize_fraction_by_leading_coefficient(expr=simplified)
    except (ValueError, TypeError, ZeroDivisionError):
        # If normalization fails, continue with current simplified form
        pass
    
    simplified = _apply_simplification_pipeline(expr=simplified)
    
    # Additional step to simplify large fraction coefficients
    simplified = _simplify_large_fraction_coefficients(expr=simplified)
    
    # Final comprehensive simplification to ensure everything is clean
    simplified = sp.simplify(simplified)
    simplified = sp.cancel(simplified)
    
    simplified_sci = _convert_large_numbers_to_float(expr=simplified)
    
    # Convert to LaTeX using SymPy's latex function with all formatting options
    # Parameters:
    # - fold_short_frac=False: Don't fold short fractions
    # - mul_symbol='dot': Use \cdot for multiplication
    # - fold_frac_powers=True: Display fractional powers as exponents
    # - fold_func_brackets=True: Omit parentheses for single-argument functions
    latex_original = sp.latex(
        sympy_expr, 
        fold_short_frac=False, 
        mul_symbol='dot', 
        fold_frac_powers=True,
        fold_func_brackets=True
    )
    latex_simplified = sp.latex(
        simplified_sci, 
        fold_short_frac=False, 
        mul_symbol='dot', 
        fold_frac_powers=True,
        fold_func_brackets=True
    )
    
    # Break long equations into multiple lines for readability
    latex_original_formatted = break_latex_equation(latex_str=latex_original, max_width=120)
    latex_simplified_formatted = break_latex_equation(latex_str=latex_simplified, max_width=120)
    
    # Return both versions in LaTeX format
    return f"${latex_original_formatted}$", f"${latex_simplified_formatted}$"


def load_env_file(env_path: str = ".env") -> dict[str, str]:
    """
    Load environment variables from a .env file.
    
    Args:
        env_path: Path to the .env file
        
    Returns:
        Dictionary of environment variables
    """
    env_vars = {}
    env_file = Path(env_path)
    
    if not env_file.exists():
        print(f"Warning: Environment file {env_path} not found")
        return env_vars
    
    try:
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Parse KEY=VALUE format
                if '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
        
        print(f"Info: Loaded {len(env_vars)} environment variables from {env_path}")
        return env_vars
        
    except (IOError, OSError, PermissionError) as e:
        print(f"Warning: Error reading environment file {env_path}: {e}")
        return env_vars

