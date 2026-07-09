<!-- markdownlint-disable MD001 MD041 -->
<p align="center">
  <img alt="YuePi0 - WidowX spoon on towel" src="https://cdn.jsdelivr.net/gh/yukki3421/YuePi0@main/docs/assets/deploy_widowx_spoon.gif" width=55%>
</p>

<p align="center">
  <em>YuePi0 deployed in SimplerEnv — WidowX "put spoon on towel"</em>
</p>

<h3 align="center">
A from-scratch, study-driven re-implementation of π0 — the Physical Intelligence VLA model
</h3>

🚧 YuePi0 is a personal reproduction project. The goal is **understanding**, not benchmarks — every module is rebuilt by hand from the [π0](https://www.physicalintelligence.company/blog/pi0) paper.

---

## About

YuePi0 (月Pi0) is a hand-written re-implementation of [π0](https://www.physicalintelligence.company/blog/pi0) — Physical Intelligence's flow-matching Vision-Language-Action model. The project follows a deliberate "rebuild-from-fundamentals" pedagogy: each subsystem (RoPE, GQA, KV-Cache, SigLip, Gemma, Mixture-of-Transformers, Joint Attention, Action Expert, Flow Matching) is built in isolation, tested with `pytest`, and `allclose`-verified against the reference HuggingFace / `open-pi-zero` implementation.

The full model is now **parameter-for-parameter compatible** with `open-pi-zero` (938 tensors, identical up to naming): an `open-pi-zero` checkpoint can be remapped and loaded directly, and the complete forward pass has been **numerically verified equivalent** via a two-process parity harness (max diff ~1e-8). Reproduction is now complete across all three tracks: numerical parity (above), SimplerEnv / WidowX deployment (success rates matching `open-pi-zero`), and a Bridge-dataset training loop (VLM frozen, bf16 action expert).


## What's Implemented

**Backbone & attention primitives**

- RMSNorm and rotary positional embeddings (RoPE)
- Grouped-Query Attention with KV-Cache
- SigLip / ViT image encoder (PaliGemma vision tower)
- Gemma decoder layers (full numerical parity with HF)

**π0-specific composition**

- VLA preprocessor (multi-modal prompt + image tiling + action chunk packing)
- Mixture-of-Transformers (MoT) scaffold — VLM expert + proprio expert + action expert
- Joint attention dispatcher across experts (`getattr`-based layer routing)
- Per-expert RoPE / KV-Cache wiring
- Block-wise causal mask (VLM causal · proprio/action bidirectional within block)
- Action expert head: TimeEncoder (sinusoidal) + ActionEncoder + ActionDecoder
- Flow-matching training loss with (1−σ_min) conditional probability path
- Euler-integrator inference loop (image → proprio → action chunk)
- adaLN / adaLN-Zero adaptive normalization for the action expert (selectable via `action_expert_adaptive_mode`)

**Verification & loading**

- Numerical parity (`allclose`) against `open-pi-zero` for the full forward pass (two-process harness, max diff ~1e-8)
- Remap-and-load of pretrained `open-pi-zero` checkpoints into YuePi0

See the up-to-date checklist below.

## Architecture

<p align="center">
  <img alt="Action expert encoders and decoder flow" src="https://cdn.jsdelivr.net/gh/yukki3421/YuePi0@main/docs/assets/action_state_encoder.jpg" width=58%>
</p>

<p align="center">
  <em>Action expert encoders & decoder — time, proprio, and noisy-action inputs in; predicted flow velocity out</em>
</p>

<p align="center">
  <img alt="SigLip ViT - image encoder forward flow" src="https://cdn.jsdelivr.net/gh/yukki3421/YuePi0@main/docs/assets/vit-process-1.png" width=58%>
</p>

<p align="center">
  <em>SigLip ViT vision tower — patch + position embedding → 27 Pre-LN encoder layers → post layernorm</em>
</p>

<p align="center">
  <img alt="Three-expert joint attention - single transformer layer flow" src="https://cdn.jsdelivr.net/gh/yukki3421/YuePi0@main/docs/assets/expert-process-1.png" width=58%>
</p>

<p align="center">
  <em>One of the 18 transformer layers — three experts (VLM / proprio / action) share a single joint attention</em>
</p>

π0 stacks 18 identical transformer layers. Each layer routes the hidden state through three parallel experts — the **VLM expert** (vision-language tokens), the **proprio expert** (robot state), and the **action expert** (action tokens) — that share one joint attention instead of attending separately. The figure traces a single layer:

1. **RMSNorm** — each expert normalizes its own hidden state
2. **Q/K/V projection** — each expert computes its own query, key, and value
3. **Concat → joint attention** — the three experts' Q/K/V are concatenated along the sequence axis and attend jointly, so the action expert can see the VLM's vision/language tokens and the proprio expert's state
4. **Split** — the joint attention output is sliced back into three streams
5. **Output projection** — per-expert attention output heads
6. **Residual** — added back to the input hidden state
7. **RMSNorm → MLP → residual** — per-expert feed-forward block, then into the next layer

This shared-attention design is the core of π0's Mixture-of-Transformers: it lets the lightweight action expert (~311M params) borrow grounding from the much larger VLM backbone (~1.98B params) without a separate cross-attention bridge.

## Reproduction Progress

- [x] VLAPreProcessor
- [x] RoPE + RMSNorm
- [x] GQA + KV-Cache
- [x] ViT / SigLip
- [x] Gemma decoder
- [x] Mixture + JointModel scaffold
- [x] Action Expert (Time / Action / Proprio encoders + decoder)
- [x] Flow-Matching loss & sampler (σ_min schedule + Euler integrator)
- [x] End-to-end forward / inference (`PiZero.forward` + `PiZero.infer_action`)
- [x] adaLN / adaLN-Zero adaptive mode for action expert
- [x] Numerical parity vs `open-pi-zero` (full forward, diff ~1e-8)
- [x] Load remapped `open-pi-zero` checkpoint into YuePi0
- [x] SimplerEnv / WidowX deployment loop
- [x] KV-cache two-phase inference (prefill + Euler denoise)
- [x] Training loop on the Bridge dataset (VLM frozen, bf16 action expert)

## Getting Started

YuePi0 is pinned to **Python 3.10** and managed with [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/yukki3421/yuepi0.git
cd yuepi0

# Create env and install (editable, src-layout)
uv venv --python 3.10
source .venv/bin/activate
uv pip install -e .
```

Run the test suite to confirm every reproduced module matches the reference:

```bash
pytest tests/ -v
```

Per-module spot checks (each mirrors the `src/` layout):

```bash
pytest tests/model/paligemma/test_vit.py        # SigLip vision tower
pytest tests/model/paligemma/test_gemma.py      # Gemma decoder
pytest tests/model/paligemma/test_gemma_allclose.py  # numerical parity vs HF
pytest tests/model/vla/test_mixture.py          # MoT dispatcher
pytest tests/model/vla/test_joint_model.py      # joint attention across experts
pytest tests/model/vla/test_pizero.py           # end-to-end forward / inference / adaptive modes
```

## Simulation Deployment (SimplerEnv / WidowX)

`scripts/run_deploy.sh` runs YuePi0 inside [SimplerEnv](https://github.com/simpler-env/SimplerEnv) on the WidowX bridge tasks: it loads a remapped `open-pi-zero` bridge checkpoint, builds the env, and rolls out episodes using two-stage KV-cache inference (prefill VLM + proprio once, then Euler-denoise the action chunk). The env adapter (`src/agent/env_adapter/simpler.py`) handles image resize, proprio reference-frame conversion (the Bridge top-down convention), action denormalization, and the Euler→axis-angle rotation conversion that SimplerEnv expects.

The loaded weights are `open-pi-zero`'s (not YuePi0-trained), so success rates track [open-pi-zero's reported numbers](https://github.com/allenzren/open-pi-zero#eval-results).

### Prerequisites

The YuePi0 venv from [Getting Started](#getting-started) must already exist. SimplerEnv / SAPIEN / ManiSkill2 are deliberately **not** declared in `pyproject.toml` — they need editable installs of their own and pin `numpy<2`, so they are wired up by hand below.

### 1. Clone SimplerEnv (allenzren fork, with submodules)

YuePi0 reuses the same SimplerEnv fork that `open-pi-zero` uses (it ships proprio support). Clone it as a sibling of this repo so the relative install paths below line up:

```bash
cd /path/to/parent      # the directory containing YuePi0/
git clone https://github.com/allenzren/SimplerEnv --recurse-submodules
```

`--recurse-submodules` is required — it pulls in `ManiSkill2_real2sim`, which SAPIEN depends on.

### 2. Install SimplerEnv + ManiSkill2_real2sim into the YuePi0 venv

```bash
cd /path/to/YuePi0
source .venv/bin/activate
pip install -e ../SimplerEnv/ManiSkill2_real2sim
pip install -e ../SimplerEnv
```

Keep `numpy < 2.0` (the YuePi0 pin `numpy==1.26.4` already satisfies this) — newer numpy breaks SAPIEN's IK.

### 3. Download the PaliGemma backbone

```bash
cd "$TRANSFORMERS_CACHE"   # e.g. ~/.cache/huggingface/hub
git clone https://huggingface.co/google/paligemma-3b-pt-224
```

### 4. Download an `open-pi-zero` bridge checkpoint

Grab one of the bridge checkpoints from the [open-pi-zero HuggingFace repo](https://huggingface.co/allenzren/open-pi-zero), e.g. `bridge_beta_step19296_2024-12-26_22-30_42.pt`. YuePi0 remaps and loads it via `load_pretrained_pizero`.

### 5. Run

`scripts/run_deploy.sh` sets three env vars and calls `scripts/deploy_simpler.py`. Edit the paths in it to match your machine:

- `PIZERO_CKPT` — path to the downloaded `.pt` checkpoint
- `TRANSFORMERS_CACHE` — HF cache dir holding `paligemma-3b-pt-224/`
- `CUDA_VISIBLE_DEVICES` — which GPU to use

Then:

```bash
bash scripts/run_deploy.sh widowx_put_eggplant_in_basket
```

### Available tasks (bridge / WidowX)

- `widowx_carrot_on_plate` — put carrot on plate
- `widowx_put_eggplant_in_basket` — put eggplant in basket
- `widowx_spoon_on_towel` — put spoon on towel
- `widowx_stack_cube` — stack cube (hardest of the four)

Rollout videos for the first few episodes are written to `result/video/`, and per-episode success + the aggregate success rate are printed to stdout.

### Gotchas

- **Always invoke `.venv/bin/python` directly, never `uv run`.** `uv run` reinstalls `setuptools`, which breaks `sapien`'s `pkg_resources` import at runtime. `run_deploy.sh` already calls `.venv/bin/python` for you — just don't bypass the script with `uv run`.
- **Run from the repo root.** `config/yuepi0.yaml`, `config/deploy_simpler.yaml`, and `config/bridge_statistics.json` are loaded by relative path (the script `cd`s to the repo root itself, so this is only a concern if you invoke `deploy_simpler.py` by hand).
- **GPU choice.** Edit `CUDA_VISIBLE_DEVICES` in `scripts/run_deploy.sh` to point at a free GPU.

## Repository Layout

```
src/model/
├── kvcache.py                  # simple per-layer KV cache
├── load_pretrained.py          # remap + load open-pi-zero checkpoints
├── paligemma/                  # vision-language backbone
│   ├── vit.py                  # SigLip ViT
│   ├── modules.py              # RMSNorm, RoPE, GQA, MLP
│   └── gemma.py                # decoder layer + full Gemma model
└── vla/                        # π0-specific composition
    ├── processing.py           # VLA preprocessor
    ├── mixture.py              # Mixture-of-Transformers expert wrapper
    ├── joint_model.py          # cross-expert joint attention
    └── yuepi0.py               # top-level model

src/agent/                      # training + deployment
│   ├── train.py                # Bridge training entry (VLM frozen, bf16)
│   └── env_adapter/            # SimplerEnv / WidowX adapters
src/data/                       # Bridge / fake dataset loaders
src/utils/geometry.py           # rotation / pose math for deployment

scripts/                        # parity harness, inspection, deploy
docs/                           # 中文学习笔记 (one per module)
tests/                          # mirrors src/ layout
config/yuepi0.yaml              # the single canonical model config
```

## Design Notes

The `docs/` directory carries Chinese-language study notes written alongside each module — they explain *why* the code looks the way it does, not just what it does:

**Primitives & backbone**

- [`rope_notes.md`](docs/rope_notes.md), [`rmsnorm_notes.md`](docs/rmsnorm_notes.md)
- [`gqa_notes.md`](docs/gqa_notes.md), [`kv_cache_notes.md`](docs/kv_cache_notes.md)
- [`vit_siglip_notes.md`](docs/vit_siglip_notes.md), [`paligemma_embedder_notes.md`](docs/paligemma_embedder_notes.md), [`gemma_notes.md`](docs/gemma_notes.md)

**π0 composition**

- [`processing_notes.md`](docs/processing_notes.md)
- [`mixture_notes.md`](docs/mixture_notes.md), [`joint_model_notes.md`](docs/joint_model_notes.md), [`dispatcher_notes.md`](docs/dispatcher_notes.md)
- [`flow_matching_and_sinusoidal_notes.md`](docs/flow_matching_and_sinusoidal_notes.md), [`pizero_inference_and_adaln_notes.md`](docs/pizero_inference_and_adaln_notes.md)

**Weights, parity & training**

- [`paligemma_weight_loading_notes.md`](docs/paligemma_weight_loading_notes.md), [`parity_test_notes.md`](docs/parity_test_notes.md)
- [`bridge_dataset_notes.md`](docs/bridge_dataset_notes.md), [`phase1_paligemma_bf16_training_notes.md`](docs/phase1_paligemma_bf16_training_notes.md)

## Acknowledgements

YuePi0 stands on the shoulders of:

- [Physical Intelligence](https://www.physicalintelligence.company/) — original π0 paper and weights
- [`allenzren/open-pi-zero`](https://github.com/allenzren/open-pi-zero) — the reference open-source PyTorch port that this repo is benchmarked against
- [`google/paligemma`](https://huggingface.co/google/paligemma-3b-pt-224) — the VLM backbone
- [`huggingface/transformers`](https://github.com/huggingface/transformers) — for `allclose` ground truth


## Contact

This is a one-person study project. For questions, open an issue on [GitHub](https://github.com/yukki3421/yuepi0/issues).
