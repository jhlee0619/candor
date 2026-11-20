# CANDOR: Contrastive Self-Denoising Rewards for Label-Free Text-to-Image Post-Training

Junhyeok Lee<sup>1</sup>, Byung-Hoon Kim<sup>2</sup>, Kyu Sung Choi<sup>1</sup><br>
<sup>1</sup>Seoul National University, <sup>2</sup>EverEx

[Paper](https://arxiv.org/abs/2611.25457) · [Project page](https://jhlee0619.github.io/candor/) · [Adapters](https://huggingface.co/jhlee0619/candor)

CANDOR rewards a text-to-image flow-matching model by how much the prompt helps it denoise its own sample: the log-ratio of unconditional to conditional denoising errors under shared noise probes, used as the only GRPO reward.

## Setup

```
git clone https://github.com/jhlee0619/candor && cd candor
pip install -r requirements.txt    # Python 3.10
export PYTHONPATH=$PWD
```

Run every command from the repository root.

## Training (8 GPUs)

```
accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=8 \
    scripts/train_sd3_self.py --config config/solace.py:general_ocr_sd3_8gpu_d_paper
```

To continue from a Flow-GRPO adapter, add `--flow_lora <adapter> --config.save_dir=<folder>`.

## Evaluation

```
python eval/gen.py --model candor --task ocr --lora logs/ocr/sd3_d_paper_8gpu/checkpoints/checkpoint-2000/lora
python eval/score.py --model candor --task ocr
```

`--task` is `ocr`, `drawbench` or `geneval` for `gen.py`, and `ocr` or `drawbench` for `score.py`. GenEval images are scored with the [GenEval code](https://github.com/djghosh13/geneval) at commit af4902f.
The DrawBench prompts and the aesthetic predictor weights are not included; copy them from [SOLACE](https://github.com/wookiekim/SOLACE) at commit 3c9b4e4 (`dataset/drawbench/test.txt`, `solace/assets/sac+logos+ava1-l14-linearMSE.pth`).

## Trained adapters

```
hf download jhlee0619/candor --local-dir adapters
python eval/gen.py --model candor --task ocr --lora adapters/candor_sd35m
```

`candor_on_flowgrpo_{geneval,text,pickscore}` are continued from `jieliu/SD3.5M-FlowGRPO-{GenEval,Text,PickScore}`; pass that adapter as `--flow_lora`.
**Powered by Stability AI.** The adapters are derived from Stable Diffusion 3.5 Medium and are released under the Stability AI Community License (see the model card).

## License

The code is under the MIT License (`LICENSE`); see `THIRD_PARTY.md` for third-party code.

## Citation

```bibtex
@article{lee2026candor,
  title   = {{CANDOR}: Contrastive Self-Denoising Rewards for Label-Free Text-to-Image Post-Training},
  author  = {Lee, Junhyeok and Kim, Byung-Hoon and Choi, Kyu Sung},
  journal = {arXiv preprint arXiv:2611.25457},
  year    = {2026}
}
```
