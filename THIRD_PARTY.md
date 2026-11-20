# Third-party code

`solace/`, `config/`, `scripts/` and `dataset/` are the files of [SOLACE](https://github.com/wookiekim/SOLACE) at commit 3c9b4e4 that CANDOR's training and evaluation use. SOLACE is built on Flow-GRPO and carries its MIT license notice, "Copyright (c) 2025 Jie Liu" (`LICENSES/MIT-SOLACE.txt`).
- We changed two of these files. `scripts/train_sd3_self.py` gets the CANDOR reward (`candor/reward.py`), the timestep weight and trained-step suffix, activation checkpointing, `--flow_lora` and a stop step. `config/solace.py` keeps only the OCR config and adds the CANDOR config.
- The other files are unchanged. The functions of `solace/rewards.py` and `solace/prompts.py` that use files not included here are not used.
- `dataset/ocr/` holds the OCR prompts of Flow-GRPO, and `dataset/geneval/test_metadata_full.jsonl` the GenEval prompts (MIT).

Not included, and used under the terms of their providers: the models and adapters named in `README.md`, the DrawBench prompts, the aesthetic predictor weights, and GenEval (commit af4902f) with mmdetection 2.28.2.
