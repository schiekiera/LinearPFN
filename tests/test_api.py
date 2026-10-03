"""The user-facing API (linearpfn.api, linearpfn.hub) on a tiny random network.

No network access and no released weights: a one-layer model with random
weights is written in the release format and loaded back, so these tests
check the plumbing (loading, preprocessing, guards, effect order, scales,
the predictive, the download cache), not the quality of the posterior.
"""

from __future__ import annotations

import hashlib
import itertools
import warnings

import numpy as np
import pytest
import torch

from linearpfn import hub
from linearpfn.api import LinearPFN, split_rows
from linearpfn.train import build_model, make_prior_config

TINY = {"embedding_size": 32, "num_attention_heads": 4, "mlp_hidden_size": 64,
        "num_layers": 1, "pair_pooling": "y_aware", "attention_impl": "sdpa",
        "norm_placement": "pre", "coef_head": "mse", "pair_stats": False}


@pytest.fixture(scope="module")
def weights(tmp_path_factory):
    torch.manual_seed(0)
    cfg = make_prior_config({})
    model, bar = build_model(TINY, cfg.include_interactions)
    path = tmp_path_factory.mktemp("w") / "tiny.pt"
    return LinearPFN(model, bar, cfg, {"model": TINY, "data": {}}, device="cpu").save(path)


@pytest.fixture()
def m(weights):
    return LinearPFN.from_pretrained(weights, device="cpu")


def data(n=60, p=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(2.0, 3.0, size=(n, p))
    y = 10 + 2 * X[:, 0] - X[:, 1] + 0.5 * X[:, 0] * X[:, 1] + rng.normal(size=n)
    return X, y


def test_weights_file_is_plain(weights):
    state = torch.load(weights, map_location="cpu", weights_only=True)
    assert state["format_version"] == 1 and set(state) == {"model", "config", "format_version"}


def test_shapes_and_effect_order(m):
    X, y = data(p=4)
    r = m.fit(X, y, feature_names=["a", "b", "c", "d"])
    pairs = list(itertools.combinations(range(4), 2))
    assert r.effect_names == ["a", "b", "c", "d"] + [f"{'abcd'[i]}:{'abcd'[j]}" for i, j in pairs]
    d = 4 + len(pairs)
    assert r.pip.shape == r.coef.shape == r.coef_mpm.shape == (d,)
    assert ((r.pip > 0) & (r.pip < 1)).all()
    np.testing.assert_array_equal(r.selected, r.pip > 0.5)
    np.testing.assert_array_equal(r.coef_mpm, np.where(r.pip > 0.5, r.coef, 0.0))
    assert r.coef_source == "head:mse" and r.n_context == 60
    assert "effect" in r.summary() and len(r.summary().splitlines()) == d + 3


def test_dataframe_like_names(m):
    class Frame:  # duck-typed: anything with .columns and __array__
        def __init__(self, a):
            self.a, self.columns = a, ["u", "v", "w"]

        def __array__(self, dtype=None, copy=None):
            return self.a if dtype is None else self.a.astype(dtype)

    X, y = data(p=3)
    assert m.fit(Frame(X), y).effect_names[:3] == ["u", "v", "w"]


def test_deterministic_and_scale_invariant(m):
    X, y = data()
    r1, r2 = m.fit(X, y), m.fit(X * 7 + 3, y * 0.5 - 1)
    np.testing.assert_allclose(r1.pip, r2.pip, atol=1e-5)
    np.testing.assert_allclose(r1.coef, r2.coef, atol=1e-5)


def test_guards(m):
    X, y = data()
    with pytest.raises(ValueError, match="p = 1"):
        m.fit(X[:, :1], y)
    with pytest.raises(ValueError, match="p = 31"):
        m.fit(np.random.default_rng(0).normal(size=(60, 31)), y)
    with pytest.raises(ValueError, match="at least 10"):
        m.fit(X[:9], y[:9])
    with pytest.warns(UserWarning, match="n >= 20"):
        m.fit(X[:15], y[:15])
    Xn = X.copy()
    Xn[3, 1] = np.nan
    with pytest.raises(ValueError, match="missing data"):
        m.fit(Xn, y)
    with pytest.raises(TypeError, match="numeric"):
        m.fit(np.array([["a", "b"]] * 20), y[:20])
    Xc = X.copy()
    Xc[:, 2] = 1.0
    with pytest.raises(ValueError, match=r"constant predictor.*x3"):
        m.fit(Xc, y)
    with pytest.raises(ValueError, match="y is constant"):
        m.fit(X, np.ones(len(y)))
    with pytest.raises(ValueError, match="length 60"):
        m.fit(X, y[:50])
    with pytest.raises(RuntimeError, match="before predict"):
        LinearPFN(m.model, m.bar, m.prior_cfg, m.config, device="cpu").predict(X)


def test_max_context_subsample(m):
    X, y = data(n=80)
    with pytest.warns(UserWarning, match="max_context"):
        r1 = m.fit(X, y, max_context=50, seed=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r2 = m.fit(X, y, max_context=50, seed=3)
        r3 = m.fit(X, y, max_context=50, seed=4)
    assert r1.n_context == 50
    np.testing.assert_array_equal(r1.pip, r2.pip)
    assert not np.array_equal(r1.pip, r3.pip)


def test_holdout_split_rule(m):
    X, y = data(n=60)
    r = m.fit(X, y, holdout=0.2, seed=11)
    ctx, q = split_rows(60, 11)
    assert len(q) == 12 and r.heldout["n_query"] == 12
    np.testing.assert_array_equal(r.heldout["query_rows"], q)
    assert np.isfinite(r.heldout["nll"]) and r.heldout["rmse"] > 0
    assert r.heldout["rmse_raw"] == pytest.approx(r.heldout["rmse"] * y[ctx].std())
    assert split_rows(20, 0)[1].size == 5  # at least five query rows


@pytest.mark.parametrize("centered", [True, False])
def test_coef_raw_reproduces_the_standardized_surface(m, centered):
    X, y = data()
    r = m.fit(X, y)
    Z = (X - r.x_mean) / r.x_sd
    zi = np.column_stack([Z[:, i] * Z[:, j] for i, j in r.pairs])
    eta_std = r.y_mean + r.y_sd * (r.intercept + np.c_[Z, zi] @ r.coef)
    B = X - r.x_mean if centered else X
    bi = np.column_stack([B[:, i] * B[:, j] for i, j in r.pairs])
    b0, b = r.coef_raw(centered=centered)
    np.testing.assert_allclose(b0 + np.c_[B, bi] @ b, eta_std, rtol=1e-10, atol=1e-8)
    b0m, bm = r.coef_raw("mpm", centered=True)
    assert np.all(bm[~r.selected] == 0)


def test_probe_source(m):
    X, y = data()
    r = m.fit(X, y, coef_source="probe")
    assert r.coef_source == "probe" and r.coef.shape == (10,)


def test_predict(m):
    X, y = data()
    m.fit(X, y)
    mean = m.predict(X[:5])
    q = m.predict(X[:5], kind="quantiles", levels=(0.1, 0.5, 0.9))
    assert mean.shape == (5,) and q.shape == (5, 3)
    assert (np.diff(q, axis=1) >= 0).all()
    dist = m.predict(X[:5], kind="distribution")
    np.testing.assert_allclose(dist.mean(), mean)
    c = dist.cdf(q[:, 1])
    np.testing.assert_allclose(c, 0.5, atol=1e-3)
    # NLL in y units = standardized NLL + log(sd): shift and scale y, NLL moves by log(10)
    m.fit(X, 10 * y + 4)
    nll10 = m.predict(X[:5], kind="distribution").nll(10 * y[:5] + 4)
    np.testing.assert_allclose(nll10, dist.nll(y[:5]) + np.log(10), atol=1e-4)
    with pytest.raises(ValueError, match="columns"):
        m.predict(X[:5, :3])


def test_hub_download_verifies_and_caches(tmp_path, monkeypatch, weights):
    repo = tmp_path / "hub" / "org" / "LinearPFN" / "resolve" / "r1"
    repo.mkdir(parents=True)
    (repo / "w.pt").write_bytes(weights.read_bytes())
    sha = hashlib.sha256(weights.read_bytes()).hexdigest()
    monkeypatch.setenv("HF_ENDPOINT", (tmp_path / "hub").as_uri())
    cache = tmp_path / "cache"
    path = hub.download("org/LinearPFN", "w.pt", "r1", sha256=sha, cache=cache,
                        backend="urllib")
    assert path.read_bytes() == weights.read_bytes()
    assert path.parent == cache / "org--LinearPFN" / "r1"
    (repo / "w.pt").unlink()  # a cache hit does not touch the endpoint
    assert hub.download("org/LinearPFN", "w.pt", "r1", sha256=sha, cache=cache,
                        backend="urllib") == path
    (repo / "w.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA-256"):
        hub.download("org/LinearPFN", "w.pt", "r1", sha256=sha, cache=tmp_path / "c2",
                     backend="urllib")
    assert not list((tmp_path / "c2").rglob("*.pt*"))
