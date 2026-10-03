"""Number formatting for the manuscript — done here, never in LaTeX.

Every quantity that reaches results_macros.tex or a table passes through one
of these functions, so prose, tables and figures round the same way. The
switches (leading zero, thousands separator, digits per metric) come from
the `style` block of configs/pipeline.yaml; defaults ("0.949",
"3,648"), APA is one flag away (".949").

Negative numbers are wrapped in \\ensuremath{} so a macro renders a real
minus sign in text and in math mode alike.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

_TEX_SPECIALS = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
                 "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


@dataclass(frozen=True)
class Style:
    leading_zero: bool = True
    thousands: str = ","
    digits: dict[str, int] = field(default_factory=lambda: {
        "auc": 3, "f1": 3, "rmse": 4, "corr": 3, "ece": 3, "nll": 3,
        "pct": 1, "seconds": 2})
    p_thresholds: tuple[float, ...] = (0.001, 0.01)

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> Style:
        s = cfg.get("style", {})
        return cls(leading_zero=bool(s.get("leading_zero", True)),
                   thousands=str(s.get("thousands", ",")),
                   digits={**cls().digits, **s.get("digits", {})},
                   p_thresholds=tuple(sorted(s.get("p_thresholds", (0.001, 0.01)))))


DEFAULT = Style()


def tex_escape(s: str) -> str:
    """Escape LaTeX specials in a plain string (labels, dataset names)."""
    return "".join(_TEX_SPECIALS.get(ch, ch) for ch in s)


def _strip_leading_zero(s: str) -> str:
    if s.startswith("0."):
        return s[1:]
    if s.startswith("-0."):
        return "-" + s[2:]
    return s


def _wrap_negative(s: str) -> str:
    return f"\\ensuremath{{{s}}}" if s.startswith("-") else s


def fmt_fixed(x: float, digits: int, style: Style = DEFAULT) -> str:
    """Fixed decimals; leading-zero policy; -0.000 collapses to 0.000."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    s = f"{float(x):.{digits}f}"
    if s.startswith("-") and float(s) == 0.0:
        s = s[1:]
    if not style.leading_zero:
        s = _strip_leading_zero(s)
    return _wrap_negative(s)


def fmt_auc(x: float, style: Style = DEFAULT) -> str:
    return fmt_fixed(x, style.digits["auc"], style)


def fmt_f1(x: float, style: Style = DEFAULT) -> str:
    return fmt_fixed(x, style.digits["f1"], style)


def fmt_rmse(x: float, style: Style = DEFAULT) -> str:
    return fmt_fixed(x, style.digits["rmse"], style)


def fmt_seconds(x: float, style: Style = DEFAULT) -> str:
    return fmt_fixed(x, style.digits["seconds"], style)


# `auc` governs the bounded selection scores (AUC, F1, precision, recall).
# ECE, correlation and NLL are separate settings because they are different
# quantities: an ECE of 0.002 and a correlation of 0.993 both lose their meaning
# at the two decimals the bounded selection scores use.
def fmt_corr(x: float, style: Style = DEFAULT) -> str:
    return fmt_fixed(x, style.digits["corr"], style)


def fmt_ece(x: float, style: Style = DEFAULT) -> str:
    return fmt_fixed(x, style.digits["ece"], style)


def fmt_nll(x: float, style: Style = DEFAULT) -> str:
    return fmt_fixed(x, style.digits["nll"], style)


def fmt_signed(x: float, digits: int, style: Style = DEFAULT) -> str:
    """Explicit sign for differences and gaps: +0.011 / -0.006."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    s = fmt_fixed(abs(float(x)), digits, style)
    if round(float(x), digits) == 0.0:
        return s
    return f"+{s}" if x > 0 else f"\\ensuremath{{-{s}}}"


def fmt_int(n: int | float, style: Style = DEFAULT) -> str:
    """Integer with a thousands separator: 3,648 (or 3\\,648)."""
    n = int(round(float(n)))
    body = f"{abs(n):,}".replace(",", style.thousands)
    return _wrap_negative(("-" if n < 0 else "") + body)


def fmt_pct(x: float, style: Style = DEFAULT, digits: int | None = None) -> str:
    """Fraction in [0, 1] (or a percentage if `x` > 1) -> '12.3\\%'."""
    d = style.digits["pct"] if digits is None else digits
    val = float(x) * 100.0 if abs(float(x)) <= 1.0 else float(x)
    return fmt_fixed(val, d, Style(leading_zero=True, thousands=style.thousands,
                                   digits=style.digits)) + r"\%"


def fmt_sig(x: float, sig: int = 3, style: Style = DEFAULT) -> str:
    """`sig` significant figures, fixed notation (0.00435, 25.3, 1,234)."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    x = float(x)
    if x == 0.0:
        return fmt_fixed(0.0, sig - 1, style)
    exp = math.floor(math.log10(abs(x)))
    decimals = max(sig - 1 - exp, 0)
    return fmt_fixed(round(x, decimals), decimals, style)


def fmt_sci(x: float, sig: int = 2) -> str:
    """Scientific notation as math: 3.5e-7 -> \\ensuremath{3.5\\times10^{-7}}."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    x = float(x)
    if x == 0.0:
        return "0"
    exp = math.floor(math.log10(abs(x)))
    mant = x / 10 ** exp
    mant_s = f"{mant:.{sig - 1}f}"
    if mant_s.startswith("10"):  # rounding overflow
        mant_s, exp = f"{1.0:.{sig - 1}f}", exp + 1
    return f"\\ensuremath{{{mant_s}\\times10^{{{exp}}}}}"


def fmt_p(p: float, style: Style = DEFAULT) -> str:
    """Relational p-value string, so the prose writes `$p \\pval$`:
    '< 0.001', '< 0.01', '= 0.023' (leading zero per style)."""
    p = float(p)
    for thr in style.p_thresholds:
        if p < thr:
            s = f"{thr:g}"
            return "< " + (s if style.leading_zero else _strip_leading_zero(s))
    return "= " + fmt_fixed(p, 3, style)


def fmt_big(n: float) -> str:
    """Compact magnitude for model cards: 25.3M, 1.2B, 812k."""
    n = float(n)
    for unit, div in (("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(n) >= div:
            v = n / div
            return f"{v:.1f}{unit}" if v < 100 else f"{v:.0f}{unit}"
    return f"{n:.0f}"


_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
          "ten", "eleven", "twelve")


def num_word(n: int) -> str:
    """Small counts as words for prose ('two GPUs'); digits beyond twelve."""
    n = int(n)
    return _WORDS[n] if 0 <= n < len(_WORDS) else str(n)


def fmt_list(items, fmt=str) -> str:
    """['32', '1,024'] -> '32 and 1,024'; three or more -> 'a, b and c'."""
    parts = [fmt(x) for x in items]
    if len(parts) <= 1:
        return "".join(parts)
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def times_word(n: int) -> str:
    """How often something happened, for prose: 'once', 'twice', 'three times'."""
    n = int(n)
    return {1: "once", 2: "twice"}.get(n, f"{num_word(n)} times")


def fmt_hours(h: float) -> str:
    """Hours -> '10.6 d' / '7.3 h' / '42 min'."""
    h = float(h)
    if h >= 48:
        return f"{h / 24:.1f} d"
    if h >= 1:
        return f"{h:.1f} h"
    return f"{h * 60:.0f} min"


def yes_no(flag: bool) -> str:
    return "yes" if flag else "no"
