# Score generated images with the SOLACE reward models (+ HPSv2.1). Writes <out>/<model>/<task>_scores.json.
import argparse, glob, json, os
import numpy as np
import torch
from PIL import Image
import solace.rewards as R

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--task", choices=["ocr", "drawbench"], required=True)
p.add_argument("--out", default="outputs")
p.add_argument("--bs", type=int, default=32)
a = p.parse_args()

f = {"ocr": "dataset/ocr/test_full.txt", "drawbench": "dataset/drawbench/test.txt"}[a.task]
prompts = [l.strip() for l in open(f) if l.strip()]
d = os.path.join(a.out, a.model, a.task)
paths = sorted(glob.glob(f"{d}/*.png"))
assert len(paths) == len(prompts), (len(paths), len(prompts))

if a.task == "ocr":
    fns = {"ocr": R.ocr_score("cuda")}
else:
    fns = {k: getattr(R, f"{k}_score")("cuda") if k != "aesthetic" else R.aesthetic_score()
           for k in ["clip", "aesthetic", "pickscore", "imagereward"]}
    import hpsv2

scores = {k: [] for k in fns}
if a.task == "drawbench":
    scores["hpsv21"] = []
for b in range(0, len(paths), a.bs):
    # float [0,1] NCHW tensor, same as train_sd3_self.py eval (output_type="pt")
    imgs = torch.from_numpy(np.stack([np.array(Image.open(x).convert("RGB")) for x in paths[b:b + a.bs]])).permute(0, 3, 1, 2).float() / 255
    ps = prompts[b:b + a.bs]
    for k, fn in fns.items():
        s, _ = fn(imgs, ps, None)
        scores[k] += [float(x) for x in (s.cpu() if torch.is_tensor(s) else s)]
    if a.task == "drawbench":
        scores["hpsv21"] += [float(hpsv2.score(os.path.abspath(x), pr, hps_version="v2.1")[0]) for x, pr in zip(paths[b:b + a.bs], ps)]
    print(f"{b + len(ps)}/{len(paths)}", flush=True)

if "pickscore" in scores:  # scorer divides by 26; paper reports raw logit
    scores["pickscore"] = [x * 26 for x in scores["pickscore"]]
if "clip" in scores:  # scorer returns 100*cos/30; paper reports cosine
    scores["clip"] = [x * 0.3 for x in scores["clip"]]
summary = {k: float(np.mean(v)) for k, v in scores.items()}
print(a.model, a.task, summary)
json.dump({"summary": summary, "per_sample": scores}, open(os.path.join(a.out, a.model, f"{a.task}_scores.json"), "w"))
