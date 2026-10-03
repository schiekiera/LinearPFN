"""Method registry for the benchmark harness.

Python methods are always available; R methods appear only when Rscript and
the required package are installed — `available_methods` says which and WHY
loudly, so a missing method is never a silent hole in the report.
"""

from __future__ import annotations

import shutil
import subprocess

from bench.harness import Method
from bench.methods.lasso import lasso_raw
from bench.methods.linearpfn_slot import linearpfn_slot
from bench.methods.pyref import exact_reference
from bench.methods.rwrap import r_method
from bench.methods.stability import stability_cpss

_R_METHODS = {
    "glinternet": "glinternet",
    "hierNet": "hierNet",
    "susie": "susieR",
}

PY_METHODS: dict[str, Method] = {
    "lasso": lasso_raw,
    "stability": stability_cpss,
    "exact": exact_reference,
    "linearpfn": linearpfn_slot,
}

# Tier-2 membership (probability quality) and default p caps live here so the
# report and the runner share one source of truth.
TIER2_METHODS = ("linearpfn", "susie", "stability", "exact")
DEFAULT_P_CAPS = {"exact": 5}


def _r_package_ok(pkg: str) -> bool:
    if shutil.which("Rscript") is None:
        return False
    probe = subprocess.run(
        ["Rscript", "-e", f"cat(requireNamespace('{pkg}', quietly=TRUE))"],
        capture_output=True, text=True, timeout=120,
    )
    return probe.stdout.strip().endswith("TRUE")


def available_methods(include_slot: bool = False) -> tuple[dict[str, Method], list[str]]:
    """(registry, loud notes about anything missing)."""
    methods = dict(PY_METHODS)
    notes: list[str] = []
    if not include_slot:
        methods.pop("linearpfn")
        notes.append("linearpfn slot EXCLUDED (no blessed checkpoint yet)")
    for name, pkg in _R_METHODS.items():
        if _r_package_ok(pkg):
            methods[name] = r_method(name)
        else:
            notes.append(
                f"R method '{name}' MISSING (package {pkg} not installed — "
                "run Rscript bench/rscripts/install_deps.R)"
            )
    return methods, notes
