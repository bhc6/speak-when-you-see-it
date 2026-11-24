"""Ego-MC-Bench (qualcomm/qualcomm-interactive-cooking-dataset-ego-mistake-corrections): download each video and
extract 2 fps frames, saved already resized to 640x360 with PIL nearest (the baseline's own load_frames_into_array
resize), then delete the MP4. Frames: <root>/frames_2fps/<video_id>/frame_%06d.jpg + .complete marker.
python mc_prepare.py ROOT [workers]"""
import json, os, sys
from multiprocessing import Pool
import cv2
from huggingface_hub import hf_hub_download
from PIL import Image

REPO = "qualcomm/qualcomm-interactive-cooking-dataset-ego-mistake-corrections"
root = sys.argv[1]; nw = int(sys.argv[2]) if len(sys.argv) > 2 else 4
os.makedirs(f"{root}/frames_2fps", exist_ok=True)
ann = json.load(open(hf_hub_download(REPO, "annotations/annotations.json", repo_type="dataset")))


def one(v):
    out = f"{root}/frames_2fps/{v}"
    if os.path.exists(f"{out}/.complete"): return v, "skip"
    mp4 = hf_hub_download(REPO, f"videos/{v}.MP4", repo_type="dataset", local_dir=f"{root}/mp4")
    os.makedirs(out, exist_ok=True)
    cap = cv2.VideoCapture(mp4); step = int(round(cap.get(cv2.CAP_PROP_FPS) / 2)); i = n = 0  # as extract_frames.py
    while True:
        ok, fr = cap.read()
        if not ok: break
        if i % step == 0:
            Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)).resize((640, 360), resample=0).save(f"{out}/frame_{n:06d}.jpg", quality=95)
            n += 1
        i += 1
    cap.release(); os.remove(mp4)
    open(f"{out}/.complete", "w").write(str(n)); return v, n


if __name__ == "__main__":
    with Pool(nw) as p: r = p.map(one, [a["video_id"] for a in ann])
    print("mc frames:", r)
