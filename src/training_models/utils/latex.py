#!/usr/bin/env python3
"""
LaTeX conversion and formatting utilities for symbolic regression expressions.
"""

import sympy as sp


def _limit_to_infinity(expr: sp.Expr, symbols: list[sp.Symbol]) -> sp.Expr | None:
    """
    Compute nested limits as each symbol -> +oo using SymPy's limit.
    Returns None if any limit fails.
    """
    result = expr
    for sym in symbols:
        try:
            result = sp.limit(result, sym, sp.oo, dir="+")
        except Exception:
            return None
    return result


def _compare_term_growth(
    term_a: sp.Expr, term_b: sp.Expr, symbols: list[sp.Symbol]
) -> int:
    """
    Compare growth of two terms as all symbols -> +oo.
    Returns 1 if term_a grows faster, -1 if term_b grows faster, 0 if unknown.
    """
    try:
        ratio = sp.simplify(sp.Abs(term_a) / sp.Abs(term_b))
        limit_val = _limit_to_infinity(expr=ratio, symbols=symbols)
        if limit_val is None:
            return 0
        if limit_val in (sp.oo, sp.zoo):
            return 1
        if limit_val == 0:
            return -1
    except Exception:
        return 0
    return 0


def get_leading_coefficient(expr: sp.Expr) -> sp.Expr:
    """Extract the leading coefficient of a polynomial expression."""
    simplified_expr = sp.simplify(expr)
    symbols = list(simplified_expr.free_symbols)
    if not symbols:
        return simplified_expr

    # Get numerator and denominator
    numer, denom = sp.fraction(simplified_expr)

    # Expand numerator to get terms
    numer_expanded = sp.expand(numer)
    terms = sp.Add.make_args(numer_expanded)
    if not terms:
        return simplified_expr

    is_polynomial = all(term.is_polynomial(*symbols) for term in terms)

    if is_polynomial:
        # Polynomial path: use degree comparison
        max_degree = -1
        max_term = None
        for term in terms:
            total_degree = sum(sp.degree(term, gen) for gen in symbols)
            if total_degree > max_degree:
                max_degree = total_degree
                max_term = term
        if max_term is None:
            return simplified_expr
        leading_coeff, _ = max_term.as_coeff_Mul()
        return leading_coeff

    # Non-polynomial path: use asymptotic growth via SymPy limits
    fastest_term = terms[0]
    for term in terms[1:]:
        comparison = _compare_term_growth(
            term_a=term, term_b=fastest_term, symbols=symbols
        )
        if comparison == 1:
            fastest_term = term
    leading_coeff, _ = fastest_term.as_coeff_Mul()
    return leading_coeff


def normalize_fraction_by_leading_coefficient(expr: sp.Expr) -> sp.Expr:
    """
    Normalize a fraction by factoring out the leading coefficient.
    Handles both polynomial expressions and expressions with variable exponents.
    """
    simplified_expr = sp.simplify(expr)
    simplified_expr = sp.cancel(simplified_expr)

    # Get numerator and denominator
    numer, denom = sp.fraction(simplified_expr)
    symbols = list(numer.free_symbols)
    if not symbols:
        return simplified_expr

    # Expand numerator to get terms
    numer_expanded = sp.expand(numer)
    terms = sp.Add.make_args(numer_expanded)
    if not terms:
        return simplified_expr

    # Check if expression is polynomial (all exponents are constant)
    # If not polynomial, we can't compute degrees and must use coefficient-based approach
    is_polynomial = all(term.is_polynomial(*symbols) for term in terms)

    max_degree = -1
    max_term = None

    if is_polynomial:
        # For polynomial expressions, find term with highest degree
        for term in terms:
            total_degree = sum(sp.degree(term, gen) for gen in symbols)
            if total_degree > max_degree:
                max_degree = total_degree
                max_term = term
    else:
        # For non-polynomial expressions (e.g., with variable exponents),
        # find the first term with a non-zero numeric coefficient
        for term in terms:
            coeff, _ = term.as_coeff_Mul()
            if coeff != 0 and coeff.is_number:
                max_term = term
                break

        # If still no term found, use first term
        if max_term is None and terms:
            max_term = terms[0]

    if max_term is None:
        return simplified_expr

    # Extract leading coefficient
    leading_coeff, _ = max_term.as_coeff_Mul()
    if leading_coeff == 1 or not leading_coeff.is_number:
        return simplified_expr

    # Normalize by dividing numerator by leading coefficient
    normalized_numer = numer / leading_coeff
    scalar_factor = leading_coeff / denom

    # Return the normalized expression
    result = scalar_factor * normalized_numer
    return sp.simplify(result)


def apply_simplification_pipeline(expr: sp.Expr) -> sp.Expr:
    """Apply a comprehensive simplification pipeline using SymPy's built-in functions."""
    # Use SymPy's comprehensive simplification
    simplified = sp.simplify(expr)

    # Apply specific simplifications in order of effectiveness
    simplified = sp.cancel(simplified)  # Cancel common factors
    simplified = sp.powsimp(simplified, combine="all", force=True)  # Combine powers
    simplified = sp.together(simplified)  # Combine fractions

    # Convert to rational numbers and factor
    # simplified = sp.nsimplify(simplified, rational=True, tolerance=1e-10)
    simplified = sp.factor_terms(simplified)

    # Final cleanup
    # simplified = sp.gcd_terms(simplified)
    simplified = sp.simplify(simplified)

    return simplified


def convert_large_numbers_to_float(expr: sp.Expr) -> sp.Expr:
    """Convert large numbers to scientific notation format."""

    def replace_large_numbers(x: sp.Expr) -> sp.Expr:
        if x.is_Number and not x.is_Symbol:
            _abs_val = abs(float(x))

            # Convert to scientific notation
            return sp.Float(x, 3)
        return x

    return expr.replace(
        lambda x: x.is_Number and not x.is_Symbol, replace_large_numbers
    )


def _check_for_invalid_terms(expr: sp.Expr, context: str) -> tuple[bool, str]:
    """Check for invalid terms using SymPy's built-in has() method."""
    # Check for infinity, complex infinity, and NaN using SymPy's has() method
    invalid_terms = [sp.oo, sp.zoo, -sp.oo, sp.nan]
    if expr.has(*invalid_terms):
        error_latex = f"\\text{{ERROR: {context} contains infinity or NaN terms}}"
        return True, error_latex
    return False, ""


def _extract_fraction_parts(latex_str: str) -> tuple[str, str] | None:
    if not latex_str.startswith("\\frac{"):
        return None
    pos = 6
    depth = 1
    numerator_chars: list[str] = []
    while pos < len(latex_str) and depth > 0:
        char = latex_str[pos]
        if char == "{":
            depth += 1
            numerator_chars.append(char)
        elif char == "}":
            depth -= 1
            if depth > 0:
                numerator_chars.append(char)
        else:
            numerator_chars.append(char)
        pos += 1
    if depth != 0 or pos >= len(latex_str) or latex_str[pos] != "{":
        return None
    numerator = "".join(numerator_chars)
    pos += 1
    depth = 1
    denominator_chars: list[str] = []
    while pos < len(latex_str) and depth > 0:
        char = latex_str[pos]
        if char == "{":
            depth += 1
            denominator_chars.append(char)
        elif char == "}":
            depth -= 1
            if depth > 0:
                denominator_chars.append(char)
        else:
            denominator_chars.append(char)
        pos += 1
    if depth != 0:
        return None
    denominator = "".join(denominator_chars)
    return numerator, denominator


def _split_at_operators(expr: str) -> list[str]:
    terms: list[str] = []
    current_term: list[str] = []
    depth = 0
    paren_depth = 0
    i = 0
    while i < len(expr):
        char = expr[i]
        if char == "{":
            depth += 1
            current_term.append(char)
        elif char == "}":
            depth -= 1
            current_term.append(char)
        elif char == "(":
            paren_depth += 1
            current_term.append(char)
        elif char == ")":
            paren_depth -= 1
            current_term.append(char)
        elif char in ["+", "-"] and depth == 0 and paren_depth == 0:
            if i > 0 and expr[i - 1] not in ["e", "E", "\\"]:
                if current_term:
                    terms.append("".join(current_term).strip())
                current_term = [char, " "]
                i += 1
                continue
            else:
                current_term.append(char)
        else:
            current_term.append(char)
        i += 1
    if current_term:
        terms.append("".join(current_term).strip())
    return terms


def _break_fraction_numerator(latex_str: str) -> str:
    parts = _extract_fraction_parts(latex_str)
    if parts is None:
        return latex_str
    numerator, denominator = parts
    num_terms = _split_at_operators(numerator)
    if len(num_terms) <= 1:
        return latex_str
    aligned_lines: list[str] = []
    for i, term in enumerate(num_terms):
        term = term.strip()
        if i == 0:
            aligned_lines.append(term)
        else:
            if term.startswith("-"):
                aligned_lines.append(term)
            elif term.startswith("+"):
                aligned_lines.append(term)
            else:
                aligned_lines.append("+" + term)
    numerator_multiline = " \\\\ ".join(aligned_lines)
    numerator_aligned = f"\\begin{{aligned}}{numerator_multiline}\\end{{aligned}}"
    result = f"\\frac{{{numerator_aligned}}}{{{denominator}}}"
    return result


def _break_regular_equation(latex_str: str, max_width: int) -> str:
    terms = _split_at_operators(latex_str)
    if len(terms) <= 1:
        return latex_str
    lines: list[str] = ["\\begin{aligned}"]
    for i, term in enumerate(terms):
        term = term.strip()
        if i == 0:
            lines.append(f"    {term}")
        else:
            if not term.startswith(("+", "-")):
                term = "+ " + term
            lines.append(f"    &{term}")
    lines.append("\\end{aligned}")
    return "\n".join(lines)


def break_latex_equation(latex_str: str, max_width: int = 120) -> str:
    if len(latex_str) <= max_width:
        return latex_str
    is_fraction = latex_str.strip().startswith("\\frac")
    if is_fraction:
        return _break_fraction_numerator(latex_str)
    else:
        return _break_regular_equation(latex_str)


_GP_VAR_LATEX = {
    "N": "N",
    "D": "D",
    "lr": "lr",
    "e": "e",
    "psi": r"\psi",
    "X0": "N",
    "X1": "D",
    "X2": "lr",
    "X3": "e",
}

_GP_PRECEDENCE = {
    "atom": 100,
    "pow": 90,
    "mul": 80,
    "div": 80,
    "add": 70,
    "sub": 70,
}


def _format_number_literal(value: object) -> str:
    if isinstance(value, sp.Number):
        value = float(value)
    if isinstance(value, (int, float)):
        text = f"{value:.12g}"
        if "." in text:
            text = text.rstrip("0").rstrip(".")
            if text == "":
                text = "0"
        return text
    return str(value)


def _gp_add(a, b):
    return ("add", a, b)


def _gp_sub(a, b):
    return ("sub", a, b)


def _gp_mul(a, b):
    return ("mul", a, b)


def _gp_div(a, b):
    return ("div", a, b)


def _gp_pow(a, b):
    return ("pow", a, b)


def _gp_var(name: str) -> tuple[str, str]:
    return ("var", _GP_VAR_LATEX.get(name, name))


def _render_gp_node(node: object) -> tuple[str, int]:
    if isinstance(node, tuple):
        kind = node[0]
        if kind == "var":
            return node[1], _GP_PRECEDENCE["atom"]
        if kind == "add":
            left, lp = _render_gp_node(node[1])
            right, rp = _render_gp_node(node[2])
            return (f"\\left({left} + {right}\\right)", _GP_PRECEDENCE["add"])
        if kind == "sub":
            left, lp = _render_gp_node(node[1])
            right, rp = _render_gp_node(node[2])
            return (f"\\left({left} - {right}\\right)", _GP_PRECEDENCE["sub"])
        if kind == "mul":
            left, lp = _render_gp_node(node[1])
            right, rp = _render_gp_node(node[2])
            if lp < _GP_PRECEDENCE["mul"]:
                left = f"\\left({left}\\right)"
            if rp < _GP_PRECEDENCE["mul"]:
                right = f"\\left({right}\\right)"
            return (f"{left} \\cdot {right}", _GP_PRECEDENCE["mul"])
        if kind == "div":
            left, _ = _render_gp_node(node[1])
            right, _ = _render_gp_node(node[2])
            return (f"\\frac{{{left}}}{{{right}}}", _GP_PRECEDENCE["div"])
        if kind == "pow":
            base, bp = _render_gp_node(node[1])
            exp, _ = _render_gp_node(node[2])
            wrap_base = bp < _GP_PRECEDENCE["pow"] or "^" in base
            if wrap_base:
                base = f"\\left({base}\\right)"
            return (f"{base}^{{{exp}}}", _GP_PRECEDENCE["pow"])
    if isinstance(node, (int, float, sp.Number)):
        return _format_number_literal(node), _GP_PRECEDENCE["atom"]
    return str(node), _GP_PRECEDENCE["atom"]


def _gplearn_expression_to_latex(expr: str) -> str | None:
    env: dict[str, object] = {
        "add": _gp_add,
        "sub": _gp_sub,
        "mul": _gp_mul,
        "div": _gp_div,
        "Pow": _gp_pow,
        "power": _gp_pow,
    }
    for name in _GP_VAR_LATEX:
        env[name] = _gp_var(name)
    try:
        tree = eval(expr, {"__builtins__": {}}, env)
    except Exception:
        return None
    latex_str, _ = _render_gp_node(tree)
    return latex_str


def convert_expression_to_latex(expression: str) -> tuple[str, str, int, int, int]:
    """
    Convert gplearn expression to LaTeX format with normalization.
    Returns: (original_expr, simplified_latex, total_terms, numerator_terms, denominator_terms)
    """
    # Map gplearn tokens to desired variable names
    N, D, WD = sp.symbols("N D WD")
    namespace = {
        "N": N,
        "D": D,
        "WD": WD,
        "add": sp.Add,
        "sub": lambda a, b: sp.Add(a, -b),
        "mul": sp.Mul,
        "div": lambda a, b: sp.Mul(a, sp.Pow(b, -1)),
    }

    
    latex_original_raw = _gplearn_expression_to_latex(expr=expression)
    sympy_expr = sp.sympify(expression, locals=namespace)
    has_invalid, error_latex = _check_for_invalid_terms(
        expr=sympy_expr, context="Expression"
    )
    if has_invalid:
        return f"${error_latex}$", f"${error_latex}$", 1, 1, 0

    # Apply simplification pipeline
    simplified = apply_simplification_pipeline(expr=sympy_expr)
    has_invalid, error_latex = _check_for_invalid_terms(
        expr=simplified, context="Simplification"
    )
    if has_invalid:
        return f"${error_latex}$", f"${error_latex}$", 1, 1, 0

    # Apply normalization as final step
    normalized = normalize_fraction_by_leading_coefficient(expr=simplified)
    # Verify leading coefficient is 1 (allow for some tolerance)
    _leading_coeff = get_leading_coefficient(normalized)

    # Convert large numbers to float for readability
    normalized_sci = convert_large_numbers_to_float(expr=normalized)

    # Generate LaTeX
    if latex_original_raw is None:
        latex_original = sp.latex(
            sympy_expr,
            fold_short_frac=False,
            mul_symbol="dot",
            fold_frac_powers=True,
            fold_func_brackets=True,
        )
    else:
        latex_original = latex_original_raw
    latex_simplified = sp.latex(
        normalized_sci,
        fold_short_frac=False,
        mul_symbol="dot",
        fold_frac_powers=True,
        fold_func_brackets=True,
    )

    # Format LaTeX
    latex_original_formatted = break_latex_equation(
        latex_str=latex_original, max_width=120
    )
    latex_simplified_formatted = break_latex_equation(
        latex_str=latex_simplified, max_width=120
    )

    # Count terms
    numer, denom = sp.fraction(normalized_sci)
    numer_terms = len(sp.Add.make_args(sp.expand(numer)))
    denom_terms = len(sp.Add.make_args(sp.expand(denom)))
    total_terms = numer_terms + denom_terms

    return (
        f"${latex_original_formatted}$",
        f"${latex_simplified_formatted}$",
        total_terms,
        numer_terms,
        denom_terms,
    )
