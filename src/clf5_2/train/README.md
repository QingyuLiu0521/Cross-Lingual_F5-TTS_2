# Training

Cross-Lingual F5-TTS 2 (CLF52) is trained with **synthetic-prompt supervised fine-tuning (SFT)**: instead of using
forced alignment to split real utterances into prompt/target, a *pretrained* vanilla F5-TTS synthesizes a same-speaker
prompt for each real utterance. The synthetic prompt is concatenated with the real speech (`x1 = [x_prompt_syn; x_target]`)
and the model is fine-tuned with an infilling / OT-CFM objective, with the missing prompt transcript replaced inside the
model by ratio-based learnable prompt tokens `<P>` plus an EOS anchor `". "`. This makes training match the
transcript-free inference setting and lets a short SFT stage adapt the pretrained model.

Training therefore has **two stages**, both launched with `accelerate` from the repo root:

1. **Stage 1 — Synthetic-prompt generation** (offline): a pretrained base F5-TTS synthesizes same-speaker prompts and
   saves them to disk.
2. **Stage 2 — Synthetic-prompt SFT**: CLF52 is fine-tuned on the real–synthetic pairs.

> All commands below assume you run them from the repo root (`Cross-Lingual_F5-TTS_2/`) inside the `clf52` conda
> environment created during [installation](../../../README.md).

- Check your FFmpeg installation:
  ```bash
  ffmpeg -version
  ```
  If not found, install it first (or skip this if you know other backends are available).

## Prerequisites

- **Base checkpoint.** Stage 1 needs a *pretrained* vanilla F5-TTS base checkpoint placed at
  `ckpts/CLF52_PretrainBase/model_1250000.safetensors` (or `.pt`). This is the model that synthesizes the prompts; it is
  described by the recipe [`src/clf5_2/configs/CLF52_PretrainBase.yaml`](../configs/CLF52_PretrainBase.yaml). If the file
  is not found under `ckpts/CLF52_PretrainBase/`, Stage 1 falls back to looking in the base recipe's own `save_dir`.
- **Tokenizer / vocab.** All stages use the `pinyin` tokenizer (see `model.tokenizer` in the configs). The SFT stage
  extends the backbone vocabulary by one learnable prompt token (`text_num_embeds = vocab_size + 1`).
- **Local vocoder** (recommended). Set `model.vocoder.is_local: True` and `model.vocoder.local_path` to a local
  Vocos mel vocoder directory (24 kHz, 100 mel), e.g. `ckpts/vocos-mel-24khz/`. This is used for logged samples during
  training.

## Stage 1 — Synthetic-prompt generation

### 1. Data preparation

Stage 1 consumes a preprocessed HuggingFace-style dataset at `data/{datasets.name}_{tokenizer}_gp_inference/`
(with `raw`/ and `duration.json`). With the default config this resolves to
`data/Emilia_ZH_EN_pinyin_gp_inference/`. Each entry holds a real utterance plus the `rel_path` used to place its
same-speaker synthetic prompt.

Build it from the raw Emilia ZH/EN data (jsonl-per-shard layout, see
[the Emilia dataset card](https://huggingface.co/datasets/amphion/Emilia-Dataset/tree/fc71e07)) with the plain-python
prep scripts under `src/clf5_2/train/datasets/`. These are **not** launched with `accelerate` — run them by path from
the repo root:

```bash
# Stage-1 input dataset -> data/Emilia_ZH_EN_pinyin_gp_inference/
python src/clf5_2/train/datasets/prepare_emilia_gp_inference.py
```

> [!IMPORTANT]
> Each prep script hard-codes its input/output paths as constants in the `if __name__ == "__main__":` block at the
> **bottom of the file** (they default to a `/vepfs/...` path). Before running, edit `dataset_dir` (the raw Emilia
> ZH/EN root) — and, for the later scripts, `gen_dir` — to point at your own paths. `langs`/`tokenizer` default to
> `["ZH", "EN"]` / `pinyin`.

> If you only want to *re-pretrain the vanilla base* (rather than run the CLF52 SFT pipeline), the base recipe reads
> `data/{datasets.name}_{tokenizer}/`, which is built with `python src/clf5_2/train/datasets/prepare_emilia.py`. This
> step is **optional** and independent of Stage 1 / Stage 2 — skip it if you use a released base checkpoint.

### 2. Configure the generation driver

The generation stage is driven by [`src/clf5_2/configs/CLF52_GenPrompt.yaml`](../configs/CLF52_GenPrompt.yaml). It is
**coupled to the base recipe**: it does not describe a model itself, but points at one via `source`:

```yaml
source:
  expname: CLF52_PretrainBase   # loads src/clf5_2/configs/CLF52_PretrainBase.yaml for the model architecture
  ckptstep: 1250000             # loads ckpts/CLF52_PretrainBase/model_1250000.{safetensors,pt}
  frac_lengths_mask: [0.2, 0.4]
```

At runtime the driver loads `CLF52_PretrainBase.yaml` to build the `CFM` model (DiT backbone, 22 layers, 16 heads,
dim 1024), then loads `ckpts/CLF52_PretrainBase/model_1250000.safetensors`. Sampling settings come from the driver's
own `infer` block:

| Field | Default | Meaning |
| --- | --- | --- |
| `infer.nfe_step` | `32` | Euler ODE solve steps |
| `infer.cfg_strength` | `2.0` | classifier-free guidance strength |
| `infer.sway_sampling_coef` | `-1.0` | sway sampling coefficient |
| `datasets.batch_size_per_gpu` | `120000` | frames per GPU (`batch_size_type: frame`) |

### 3. Run generation

```bash
# setup accelerate config once, e.g. multi-gpu ddp + bf16 mixed-precision.
# written to: ~/.cache/huggingface/accelerate/default_config.yaml
accelerate config

accelerate launch src/clf5_2/train/inference_gp.py --config-name CLF52_GenPrompt
```

The base model synthesizes each same-speaker prompt via `CFM.sample_reverse` and writes a `.pt` (mel) and a `.json`
(metadata) file per utterance into `{datasets.name}_gen_fixed/` at the repo root — i.e. `Emilia_ZH_EN_gen_fixed/` with
the default config. The internal directory layout mirrors the source dataset. Generation is resumable (fixed shuffle
seed `666`), so it can be safely re-launched to fill in missing outputs.

## Stage 2 — Synthetic-prompt SFT

### 1. Data preparation

The SFT stage reads a preprocessed dataset at `data/{datasets.name}_{tokenizer}_sft/` (default:
`data/Emilia_ZH_EN_pinyin_sft/`). It contains `raw.arrow`, `duration.json` (which also stores the per-sample
`prompt_frames` list — the synthetic-prompt length from Stage 1 — used to place the ratio-based prompt tokens),
and `vocab.txt`. Build it from the Stage-1 outputs in `{datasets.name}_gen_fixed/` plus the original real speech.

Like the Stage-1 prep, these scripts are plain python (not `accelerate`), run by path from the repo root, and read
their input/output paths from constants at the **bottom of each file** — edit `dataset_dir` (raw Emilia root) and
`gen_dir` (the Stage-1 `..._gen_fixed/` output) to your paths before running.

**a) Bridge — merge Stage-1 generations back into the Emilia jsonls (required).** Stage-1 generation writes one
`.pt`/`.json` per utterance into `{datasets.name}_gen_fixed/`, but the SFT prep scripts read the `gen_text` /
`gen_len` fields from the jsonls. This step merges each original Emilia jsonl with the matching
`{datasets.name}_gen_fixed/*.json`, adding those two fields. **Without it Stage 2 cannot be built.**

```bash
# rewrites the jsonls under gen_dir in place, adding gen_text / gen_len
python src/clf5_2/train/datasets/newjsonl_gp.py
```

Set `orig_dataset_dir` (raw Emilia root) and `gen_dataset_dir` (the Stage-1 `..._gen_fixed/` output) at the **top** of
`newjsonl_gp.py` (lines 7-8) before running — unlike the `prepare_*.py` scripts, this one keeps its path constants at
the top of the file rather than in a bottom `__main__` block.

**b) Full SFT dataset -> `data/Emilia_ZH_EN_pinyin_sft/`.**

```bash
python src/clf5_2/train/datasets/prepare_emilia_gp_train.py
```

**c) (Optional) data-scale subset -> `data/Emilia_ZH_EN_pinyin_sft_hours{target_hour}/`** (the paper's B0–B3
data-scale subsets). This selects, per language, the top samples by native Emilia DNSMOS up to `target_hour`
(`500 | 1000 | 2500`, set at the bottom of the script).

```bash
python src/clf5_2/train/datasets/prepare_emilia_gp_train_filter_d.py
```

> [!IMPORTANT]
> Run order matters: the subset script **reuses the full set's vocab** (`data/Emilia_ZH_EN_pinyin_sft/vocab.txt`), so
> run `prepare_emilia_gp_train.py` (step b) **first**. To train on a subset, set `datasets.target_hour` in the SFT
> config so it resolves `data/{datasets.name}_{tokenizer}_sft_hours{target_hour}/`.

### 2. Configure the SFT recipe

The SFT recipe is [`src/clf5_2/configs/CLF52_Base.yaml`](../configs/CLF52_Base.yaml). The fields that define the
synthetic-prompt / prompt-token recipe are:

| Field | Default | Meaning |
| --- | --- | --- |
| `model.prefix_text_pad` | `True` | insert `N` prefix tokens before the target text, with `N` set from the prompt/target duration **ratio** |
| `model.prompt_token` | `True` | the prefix uses a **learnable** prompt token `<P>` (extends vocab by 1); if `False`, uses filler id `-1` |
| `model.eos_marker` | `". "` | EOS anchor inserted between the prefix tokens and the target text |
| `model.use_total_text` | `False` | must stay `False`; mutually exclusive with `prefix_text_pad` |
| `model.name` | `F5TTS_v1_Base` | run/checkpoint name |
| `model.backbone` | `DiT` | backbone (`DiT` is what CLF52 uses) |
| `datasets.name` | `Emilia_ZH_EN` | dataset name (used to resolve `data/…_sft/`) |

Training hyperparameters (matching the paper's recipe — 100K updates on 8× A100, per-GPU batch 38,400 frames,
AdamW, LR linearly warmed to 7.5e-5 over the first 20K updates then linearly decayed):

| Field | Default |
| --- | --- |
| `datasets.batch_size_per_gpu` | `38400` (frame-wise) |
| `datasets.max_samples` | `64` |
| `optim.learning_rate` | `7.5e-5` |
| `optim.num_warmup_updates` | `20000` |
| `optim.epochs` | `11` |
| `optim.grad_accumulation_steps` | `1` |
| `optim.max_grad_norm` | `1.0` |
| `ckpts.save_per_updates` | `10000` |
| `ckpts.last_per_updates` | `5000` |
| `ckpts.logger` | `wandb` |

Checkpoints are written to `ckpts.save_dir`, which defaults to
`ckpts/${model.name}_${model.mel_spec.mel_spec_type}_${model.tokenizer}_${datasets.name}_sft`.

> [!NOTE]
> The model is initialized inside `train_sft.py` as a `CFM_SFT` with `prefix_text_pad`, `prompt_token`, and
> `eos_marker` wired in from the config, so make sure these three fields describe the recipe you intend to train.

### 3. Run SFT training

```bash
accelerate launch src/clf5_2/train/train_sft.py --config-name CLF52_Base

# you can override accelerate and hydra settings on the command line, e.g.
accelerate launch --mixed_precision=bf16 src/clf5_2/train/train_sft.py \
    --config-name CLF52_Base ++datasets.batch_size_per_gpu=19200
```

The resulting model is the CLF52 checkpoint used for inference; the released weights are
`ckpts/clf52_100000.safetensors` (100K updates). See the [inference guide](../infer/README.md) to run zero-shot,
transcript-free voice cloning with it.

## W&B logging

The `wandb/` directory is created under the path where you launch training.

By default the SFT recipe sets `ckpts.logger: wandb`. If you have not logged in, W&B runs in disabled/anonymous mode.
To enable logging, either:

1. Log in once with `wandb login` (see the [W&B docs](https://docs.wandb.ai/ref/cli/wandb-login)), or
2. Set an API key programmatically (get one at https://wandb.ai/authorize):

   On macOS & Linux:
   ```bash
   export WANDB_API_KEY=<YOUR WANDB API KEY>
   ```
   On Windows:
   ```bat
   set WANDB_API_KEY=<YOUR WANDB API KEY>
   ```

To log offline when you cannot reach W&B:
```bash
export WANDB_MODE=offline
```

To disable logging entirely, set `ckpts.logger: null` in the config.

## See also

- [`../configs/`](../configs/) — `CLF52_Base.yaml` (SFT recipe), `CLF52_GenPrompt.yaml` (Stage-1 driver),
  `CLF52_PretrainBase.yaml` (base recipe referenced by Stage 1).
- [`../infer/README.md`](../infer/README.md) — transcript-free zero-shot inference.
- [`../eval/README.md`](../eval/README.md) — WER/CER, speaker similarity, and UTMOS evaluation.
- [`../../rate_pred/train/README.md`](../../rate_pred/train/README.md) — training the silence-robust Speaking Rate
  Predictor (SRP) used to predict duration at inference.
