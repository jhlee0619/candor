# Generate evaluation images from the repository root (prompt files under dataset/).
# SD3.5-M: 512px, 40 steps, CFG 7.0, negative prompt "", max_sequence_length 128.
import argparse, json, os
import torch
from diffusers import StableDiffusion3Pipeline
from peft import PeftModel

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)  # output name
p.add_argument("--lora", default=None, help="LoRA adapter to merge (local directory or model ID)")
p.add_argument("--flow_lora", default=None, help="Flow-GRPO adapter merged before --lora")
p.add_argument("--task", choices=["ocr", "drawbench", "geneval"], required=True)
p.add_argument("--shard", type=int, default=0)
p.add_argument("--nshards", type=int, default=1)
p.add_argument("--bs", type=int, default=16)
p.add_argument("--out", default="outputs")
p.add_argument("--cfg", type=float, default=7.0)
a = p.parse_args()

if a.task == "geneval":
    seen, meta = set(), []
    for l in open("dataset/geneval/test_metadata_full.jsonl"):
        m = json.loads(l)
        if m["prompt"] not in seen:
            seen.add(m["prompt"]); meta.append(m)
    # GenEval protocol: 4 samples per prompt
    jobs = [(i, s, m["prompt"]) for i, m in enumerate(meta) for s in range(4)]
else:
    f = {"ocr": "dataset/ocr/test_full.txt", "drawbench": "dataset/drawbench/test.txt"}[a.task]
    jobs = [(i, 0, l.strip()) for i, l in enumerate(open(f)) if l.strip()]

out = os.path.join(a.out, a.model, a.task)
def path(i, s):
    return os.path.join(out, f"{i:05d}", "samples", f"{s:04d}.png") if a.task == "geneval" else os.path.join(out, f"{i:05d}.png")

jobs = [j for j in jobs[a.shard::a.nshards] if not os.path.exists(path(j[0], j[1]))]
print(f"{a.model}/{a.task} shard {a.shard}: {len(jobs)} todo", flush=True)
if not jobs:
    raise SystemExit

pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16)
kw = dict(height=512, width=512, num_inference_steps=40, guidance_scale=a.cfg, max_sequence_length=128)
for lora in (a.flow_lora, a.lora):  # Flow-GRPO adapter first
    if lora:
        pipe.transformer = PeftModel.from_pretrained(pipe.transformer, lora).merge_and_unload()
pipe = pipe.to("cuda")
pipe.set_progress_bar_config(disable=True)

if a.task == "geneval":
    for i, m in enumerate(meta):
        os.makedirs(os.path.join(out, f"{i:05d}", "samples"), exist_ok=True)
        with open(os.path.join(out, f"{i:05d}", "metadata.jsonl"), "w") as fh:
            json.dump(m, fh)
else:
    os.makedirs(out, exist_ok=True)

for b in range(0, len(jobs), a.bs):
    batch = jobs[b:b + a.bs]
    # fixed per-sample seed, so that all models see identical initial noise
    gens = [torch.Generator("cuda").manual_seed(i * 4 + s) for i, s, _ in batch]
    imgs = pipe([x[2] for x in batch], negative_prompt=[""] * len(batch), generator=gens, **kw).images
    for (i, s, _), img in zip(batch, imgs):
        img.save(path(i, s))
    print(f"{b + len(batch)}/{len(jobs)}", flush=True)
