"""Download the CaptainCook4D GoPro 360p videos for one Qualcomm Interactive Cooking split, nothing else.

Uses the official downloader's own link table and download_data (ref_sys/captaincook4d_downloader, a clone of
github.com/CaptainCook4D/downloader), with the same 360p -> Hololens fallback as its download_gopro_data.py; the
official script has no per-recording filter, so this only narrows its loop to the split's video ids. Output layout
is the one the quick-start loader expects (ref_sys/ai_coach_cooking_2026): <root>/captain_cook_4d/gopro/
resolution_360p/<id>_360p.mp4.

    uv run python experiments/plan_repr/qic_download.py --plan_set main --split test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DL = ROOT / "ref_sys" / "captaincook4d_downloader"
sys.path.insert(0, str(DL))

from util import Constants, download_data  # noqa: E402

HF = (f"{os.environ.get('HF_ENDPOINT', 'https://huggingface.co').rstrip('/')}"
      "/datasets/qualcomm/qualcomm-interactive-cooking-dataset/resolve/main")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan_set", default="main", choices=["main", "advanced_planning"])
    ap.add_argument("--split", default="test", choices=["train", "validation", "test"])
    ap.add_argument("--output_dir", default=str(ROOT / "data" / "external" / "captaincook4d"))
    ap.add_argument("--ids", default=None, help="comma-separated subset of the split's video ids")
    a = ap.parse_args()
    ids = sorted(pd.read_parquet(f"{HF}/{a.plan_set}/{a.split}-00000-of-00001.parquet").video_id.unique())
    if a.ids:
        want = set(a.ids.split(",")); missing = want - set(ids)
        assert not missing, f"not in {a.plan_set}/{a.split}: {sorted(missing)}"
        ids = [i for i in ids if i in want]
    links = json.loads((DL / "metadata" / "download_links.json").read_text())
    out = Path(a.output_dir) / Constants.CAPTAIN_COOK_4D / Constants.GOPRO / Constants.RESOLUTION_360P
    urls, paths = [], []
    for rid in ids:
        rec = links.get(rid) or {}
        url = rec.get(Constants.GOPRO_RESOLUTION_360P) or rec.get(Constants.HOLOLENS_SYNC_PV_VIDEO)
        if url is None:
            print(f"No 360P video available for {rid}")
            continue
        if Constants.GOPRO_RESOLUTION_360P not in rec:
            print(f"No GoPro 360P for {rid}, using the Hololens view instead")
        urls.append(url)
        paths.append(out / f"{rid}_360p.mp4")
    print(f"{a.plan_set}/{a.split}: {len(ids)} recordings, downloading {len(urls)} files to {out}")
    download_data(urls, paths)


if __name__ == "__main__":
    main()
