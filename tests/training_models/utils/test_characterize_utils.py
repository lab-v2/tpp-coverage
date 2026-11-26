"""Characterization of `src/training_models/utils/`: coloured output, .env files and LaTeX.

Expected values are the base commit's outputs, frozen in `frozen/` (see `tests/characterize.py`).
`utils/utils.py` is an older copy of these three modules that nothing imports; it is left out.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import sympy as sp

from characterize import check_frozen
from training_models import utils as U
from training_models.utils import latex

FROZEN = Path(__file__).parent / "frozen"
PRINTERS = ["red", "blue", "green", "yellow", "purple", "cyan", "bold", "underline"]
# gplearn programs over the trainer's (N, D, WD) symbols, in the converter's input syntax.
PROGRAMS = [
    "add(div(N, D), 1.69)",
    "sub(mul(0.5, N), div(D, WD))",
    "div(add(mul(406.4, D), mul(410.7, N)), mul(N, D))",
    "add(div(1234567.0, N), div(add(D, 3.0), mul(2.0, add(N, D))))",
    "div(N, sub(D, D))",
    "mul(add(N, 1e-9), 2.5e7)",
]


def test_colored_output(capsys):
    view = {"colored": {c: U.colored_text("x", c) for c in [*PRINTERS, "BLUE", "none"]}}
    for c in PRINTERS:
        getattr(U, f"print_{c}")("msg")
    U.print_colored("msg", "purple")
    U.print_header("Title")
    U.print_header("Title", color="red", width=10)
    U.print_section("Section")
    for kind in ("success", "error", "warning", "info"):
        getattr(U, f"print_{kind}")("msg")
    view["printed"] = capsys.readouterr().out.splitlines()
    check_frozen(FROZEN / "colors.json", view)


def test_load_env_file(tmp_path, capsys):
    env = tmp_path / ".env"
    env.write_text("# comment\n\nA=1\n B = two words \nURL=http://h/?a=b=c\nNOEQUALS\n")
    view = {
        "parsed": U.load_env_file(str(env)),
        "missing": U.load_env_file(str(tmp_path / "absent.env")),
        "directory": U.load_env_file(str(tmp_path)),
        "printed": capsys.readouterr().out.replace(str(tmp_path), "<tmp>").splitlines(),
    }
    check_frozen(FROZEN / "env.json", view)


@pytest.mark.parametrize("i", range(len(PROGRAMS)))
def test_convert_expression_to_latex(i):
    check_frozen(FROZEN / f"latex_{i}.json", list(latex.convert_expression_to_latex(PROGRAMS[i])))


def test_latex_helpers():
    N, D = sp.symbols("N D")
    long_sum = " + ".join(f"{k}.5 N^{{{k}}}" for k in range(1, 40))
    # A long equation that is not a fraction raises today: break_latex_equation calls
    # _break_regular_equation without its max_width argument (pinned here; a fix changes it).
    with pytest.raises(TypeError, match="max_width"):
        latex.break_latex_equation(long_sum, max_width=60)
    view = {
        "break_regular": latex._break_regular_equation(long_sum, max_width=60),
        "break_fraction": latex.break_latex_equation(
            r"\frac{" + long_sum + "}{D + 1}", max_width=60
        ),
        "short": latex.break_latex_equation("N + D"),
        "normalized": str(latex.normalize_fraction_by_leading_coefficient((4 * N + 2) / (8 * D))),
        "leading": str(latex.get_leading_coefficient(3 * N**2 + N)),
        "large_float": str(latex.convert_large_numbers_to_float(sp.Integer(123456789) * N)),
        "invalid": list(latex._check_for_invalid_terms(sp.zoo * N, "ctx")),
        "valid": list(latex._check_for_invalid_terms(N + D, "ctx")),
    }
    check_frozen(FROZEN / "latex_helpers.json", view)
