"""On-disk cache for the exact-reference outputs of the validation panel.

The exact posterior depends only on (X, y, prior, quadrature grid), and the
validation panel's datasets are a deterministic function of (panel seed,
prior config) — so across checkpoints of the same prior the reference is
bit-identical, and recomputing it for every validation run would repeat ~1-3 h
of p=5 quadrature fits. This cache stores exactly the four per-dataset outputs
`validate._evaluate_cell` consumes: nll_exact, coef_exact, exact_pips,
comp_mass.

Staleness contract (deliberately strict): the cache file lives at a FIXED
per-panel path and stores its key components (prior-config hash, panel spec,
quadrature node counts, source hash of prior/_xgen/reference/_quadrature).
On load, any component mismatch RAISES naming the culprit — never a silent
recompute, never a stale reference. Rebuilding requires the explicit
`--rebuild-ref-cache` flag. An absent file means first build: the validation
run computes the references and saves them on completion.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from linearpfn.prior import PriorConfig

__all__ = ["ReferenceCache", "cache_components"]

_SOURCE_MODULES = ("prior", "_xgen", "reference", "_quadrature")
_FIELDS = ("nll", "mass", "coef", "pips")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def cache_components(prior_cfg: PriorConfig, val_cfg: dict) -> dict[str, str]:
    """Key components; every one must match for a cached panel to be used."""
    from linearpfn import reference

    pkg = Path(__file__).resolve().parent
    code = b"".join((pkg / f"{name}.py").read_bytes() for name in _SOURCE_MODULES)
    prior_ser = json.dumps(asdict(prior_cfg), sort_keys=True)
    return {
        "prior": _sha(prior_ser.encode()),
        "panel": f"seed={val_cfg['seed']},n_query={val_cfg['n_query']}",
        "quadrature": f"c={reference.QUAD_NODES_C},rho={reference.QUAD_NODES_RHO}",
        "code": _sha(code),
    }


class ReferenceCache:
    """Either serves a complete cached panel or records a build; never both.

    get() on a loaded cache raises for any cell/index it does not hold (a
    partial cache must be rebuilt, not silently topped up); get() during a
    build returns None and the freshly computed values are recorded via put().
    """

    def __init__(self, directory: str | Path, components: dict[str, str],
                 rebuild: bool = False) -> None:
        self.components = components
        # Prior hash in the FILENAME: different prior configurations
        # legitimately coexist as separate caches. Code/quadrature drift keeps
        # the same filename and therefore RAISES on load — that is the
        # staleness the fixed path exists to catch.
        stem = components["panel"].replace(",", "_").replace("=", "")
        self.path = Path(directory) / f"refcache_{stem}_{components['prior'][:8]}.npz"
        self._cells: dict[tuple[int, int], dict[str, np.ndarray]] = {}
        self._building: dict[tuple[int, int], list[dict]] = {}
        self.hits = 0
        self.computed = 0
        if self.path.exists() and not rebuild:
            self._load()
            print(f"reference cache: loaded {self.path} "
                  f"({sum(v['nll'].size for v in self._cells.values())} datasets)")
        else:
            reason = "rebuild requested" if self.path.exists() else "absent"
            print(f"reference cache: {reason} — building at {self.path}")

    @property
    def loaded(self) -> bool:
        return bool(self._cells)

    def _load(self) -> None:
        z = np.load(self.path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        mismatched = [k for k in self.components if meta.get(k) != self.components[k]]
        if mismatched:
            detail = "; ".join(
                f"{k}: cached {meta.get(k)!r} != current {self.components[k]!r}"
                for k in mismatched
            )
            raise ValueError(
                f"reference cache {self.path} is STALE (mismatched: {detail}). "
                "Refusing to use it or to silently recompute — rerun with "
                "--rebuild-ref-cache after confirming the change is intended."
            )
        for name in z.files:
            if name == "meta":
                continue
            field, n, p = name.rsplit("_", 2)
            self._cells.setdefault((int(n), int(p)), {})[field] = z[name]

    def get(self, n: int, p: int, i: int) -> dict | None:
        if not self.loaded:
            return None
        cell = self._cells.get((n, p))
        if cell is None:
            raise ValueError(
                f"reference cache {self.path} has no cell (n={n}, p={p}) — the "
                "panel grew since the cache was built; rerun with --rebuild-ref-cache"
            )
        if i >= cell["nll"].size:
            raise ValueError(
                f"reference cache {self.path} holds {cell['nll'].size} datasets for "
                f"cell (n={n}, p={p}) but index {i} was requested — rerun with "
                "--rebuild-ref-cache"
            )
        self.hits += 1
        return {
            "nll_exact": float(cell["nll"][i]),
            "comp_mass": float(cell["mass"][i]),
            "coef_exact": cell["coef"][i],
            "exact_pips": cell["pips"][i],
        }

    def put(self, n: int, p: int, i: int, nll_exact: float, comp_mass: float,
            coef_exact: np.ndarray, exact_pips: np.ndarray) -> None:
        rows = self._building.setdefault((n, p), [])
        assert i == len(rows), "cache build must append datasets in stream order"
        rows.append({"nll": nll_exact, "mass": comp_mass,
                     "coef": np.asarray(coef_exact, dtype=np.float64),
                     "pips": np.asarray(exact_pips, dtype=np.float64)})
        self.computed += 1

    def save(self) -> None:
        """Persist a completed build (no-op when serving a loaded cache)."""
        if self.loaded or not self._building:
            return
        arrays: dict[str, np.ndarray] = {}
        for (n, p), rows in self._building.items():
            arrays[f"nll_{n}_{p}"] = np.array([r["nll"] for r in rows], dtype=np.float64)
            arrays[f"mass_{n}_{p}"] = np.array([r["mass"] for r in rows], dtype=np.float64)
            arrays[f"coef_{n}_{p}"] = np.stack([r["coef"] for r in rows])
            arrays[f"pips_{n}_{p}"] = np.stack([r["pips"] for r in rows])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        meta = np.array(json.dumps(self.components))
        tmp = self.path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, meta=meta, **arrays)
        tmp.replace(self.path)
        print(f"reference cache: saved {self.computed} datasets to {self.path}")
