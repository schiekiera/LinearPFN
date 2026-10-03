"""Stage 50 — the numbers of the real-design sampling pipeline (data_prep/real_designs).

    python scripts/50_real_x_numbers.py

Index size, included packages, frame, downloads, eligibility, exclusions
by stage and reason, the 4 x 4 cell table (selected / eligible), the
selection's max|r|, seed and R version — all from the pipeline's finished
CSVs; plus the verification receipt (work/VERIFIED.txt) when 09_verify.R
has written one. Output: reports/paper/real_x.json.
"""

from __future__ import annotations

from _bootstrap import CONFIG, NUM, P, paths, store
from pipeline import realx


def main() -> int:
    rx = P["real_data_x"]
    payload = realx.numbers(rx)
    receipt = rx / "work" / "VERIFIED.txt"
    payload["verify_receipt"] = receipt.read_text().strip() if receipt.is_file() else None
    inputs = [CONFIG, rx / "selected_datasets.csv", rx / "eligible_designs.csv",
              rx / "exclusions.csv", rx / "MANIFEST.csv",
              rx / "work" / "00_datasets.csv", rx / "work" / "02_frame.csv",
              rx / "work" / "07_cells.csv", rx / "00_config.R", rx / "10_deliver_eligible.R",
              rx / "MANIFEST_sessionInfo.txt",
              receipt]  # recorded as MISSING until 09_verify.R writes it
    out = store.write_json(NUM / "real_x.json", payload, inputs, stage="50_real_x_numbers")
    s = payload["selected"]
    print(f"index {payload['index']['datasets']}, frame {payload['frame']}, eligible "
          f"{payload['eligible']}, selected {s['datasets']} from {s['packages']} packages\n"
          f"-> {paths.rel(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
