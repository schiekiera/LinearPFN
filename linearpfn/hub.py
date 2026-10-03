"""Download of the pretrained weights from the Hugging Face Hub, with SHA-256 check.

`huggingface_hub` is optional. Without it the file comes over plain HTTPS from
`{HF_ENDPOINT or https://huggingface.co}/<repo>/resolve/<revision>/<file>`. A
private repository needs a token (argument or the HF_TOKEN environment variable).
Files are cached under `~/.cache/linearpfn` (or LINEARPFN_CACHE) and verified
against their SHA-256 on download and on every cache hit.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ID = "schiekiera/LinearPFN"
FILENAME = "linearpfn_strong.pt"
REVISION = "main"
SHA256: str | None = "4ed4e0bd81035c27cdd0902db9f6068612342a3de4a0692c3a5cdb7167bff5a9"
_CHUNK = 1 << 20


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def cache_dir() -> Path:
    env = os.environ.get("LINEARPFN_CACHE")
    return Path(env).expanduser() if env else Path.home() / ".cache" / "linearpfn"


def _check(path: Path, sha256: str | None) -> None:
    if sha256 is not None and (got := sha256_of(path)) != sha256:
        raise ValueError(f"{path.name}: SHA-256 {got} does not match the expected {sha256}")


def _urllib_download(url: str, dst: Path, token: str | None, sha256: str | None) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "linearpfn"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dst.parent, prefix=dst.name + ".", suffix=".part")
    h = hashlib.sha256()
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(req) as resp:
            for block in iter(lambda: resp.read(_CHUNK), b""):
                out.write(block)
                h.update(block)
        if sha256 is not None and h.hexdigest() != sha256:
            raise ValueError(f"{dst.name}: SHA-256 {h.hexdigest()} of the download does not "
                             f"match the expected {sha256}")
        os.chmod(tmp, 0o644)
        os.replace(tmp, dst)
    except urllib.error.HTTPError as exc:
        hint = " (a private repository needs a token: pass token= or set HF_TOKEN)" \
            if exc.code in (401, 403, 404) else ""
        raise RuntimeError(f"download of {url} failed: HTTP {exc.code}{hint}") from exc
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def download(repo_id: str = REPO_ID, filename: str = FILENAME, revision: str = REVISION,
             sha256: str | None = None, token: str | None = None,
             cache: str | Path | None = None, backend: str = "auto") -> Path:
    """Local path of `filename` from the Hub repository `repo_id` at `revision`.

    backend: "auto" uses huggingface_hub when it is installed and HF_ENDPOINT is not a
    file:// URL, else "urllib"; "hub" or "urllib" force one.
    """
    token = token or os.environ.get("HF_TOKEN") or None
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    if backend == "auto":
        try:
            import huggingface_hub  # noqa: F401
            backend = "urllib" if endpoint.startswith("file:") else "hub"
        except ImportError:
            backend = "urllib"
    if backend == "hub":
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(repo_id, filename, revision=revision, token=token))
        _check(path, sha256)
        return path
    if backend != "urllib":
        raise ValueError(f"unknown backend {backend!r}")
    dst = Path(cache).expanduser() if cache else cache_dir()
    dst = dst / repo_id.replace("/", "--") / revision / filename
    if dst.is_file():
        try:
            _check(dst, sha256)
            return dst
        except ValueError:
            dst.unlink()  # a stale or corrupt cache entry is fetched again
    url = f"{endpoint}/{repo_id}/resolve/{urllib.parse.quote(revision)}/" \
          f"{urllib.parse.quote(filename)}"
    _urllib_download(url, dst, token, sha256)
    return dst
