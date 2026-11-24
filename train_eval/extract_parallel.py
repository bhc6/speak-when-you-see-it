"""Parallel 2 fps frame extraction with the official extract_frames.py functions (process_video, save_frames_to_folder)
for the videos of one QICD split. Each finished folder gets a .complete marker (frame count), so reruns skip it and
an interrupted folder is redone. python extract_parallel.py ROOT SPLIT [workers]"""
import os, sys, json
from multiprocessing import Pool
sys.path.insert(0, os.environ["QIC_BASELINE_DIR"])
import extract_frames as E
root, split = sys.argv[1], sys.argv[2]; nw = int(sys.argv[3]) if len(sys.argv) > 3 else 16
ids = json.load(open(f"WORKSPACE/qic/results/explore/videos_{split}.json"))
src, dst = f"{root}/resolution_360p", f"{root}/resolution_360p_video_frames_2fps"


def one(v):
    mp4, out = f"{src}/{v}_360p.mp4", f"{dst}/{v}_360p"
    if os.path.exists(f"{out}/.complete"): return v, "skip"
    if not os.path.exists(mp4): return v, "missing"
    os.makedirs(out, exist_ok=True)
    fr = E.process_video(mp4, 2); E.save_frames_to_folder(out, fr)
    open(f"{out}/.complete", "w").write(str(len(fr))); return v, len(fr)


if __name__ == "__main__":
    with Pool(nw) as p: r = p.map(one, ids)
    bad = [v for v, x in r if x == "missing"]
    print(f"{split}: {len(ids)} videos, extracted {sum(isinstance(x, int) for _, x in r)}, skipped {sum(x == 'skip' for _, x in r)}, missing {bad}")
    sys.exit(1 if bad else 0)
