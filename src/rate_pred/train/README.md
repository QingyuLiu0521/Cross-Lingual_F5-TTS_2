# Speaking Rate Predictor Training

This folder trains the **silence-robust Speaking Rate Predictor (SRP)** used by CLF52 for transcript-free duration prediction. At inference the SRP takes only the mel-spectrogram of the audio prompt and predicts a syllable-level speaking rate; the target duration is then computed as `target linguistic units / predicted speaking rate`. Silence-aware augmentation makes this estimate robust to leading/trailing silence in the prompt.

The trained checkpoint corresponds to `ckpts/silence-robust-srp.safetensors` used by inference (see `src/clf5_2/infer/examples/clf52/clf52.toml`, field `srp_ckpt_file`).

## Prepare Dataset

The SRP is trained on a balanced English + Chinese subset of the [Emilia](https://huggingface.co/datasets/amphion/Emilia-Dataset) dataset. The preprocessing script reads per-book JSONL manifests (each line has `wav`, `text`, `duration`, `language`), computes a syllable-level speaking-rate class per utterance, filters bad cases, and packs the result into Arrow files.

Run the prepare script from the repo root:

```bash
python src/rate_pred/train/datasets/prepare_emilia_speed.py
```

Edit the constants at the bottom of `prepare_emilia_speed.py` to match your setup:

| Constant | Meaning |
| --- | --- |
| `DATASET_DIR` | Root of the raw Emilia data (contains `ZH/` and `EN/` subfolders of `*.jsonl`) |
| `LANGS` | Languages to include, `["ZH", "EN"]` |
| `MAX_DURATION_HOURS` | Hours per language for the train split (`500` → `500 × 2 = 1000h` total) |
| `VAL_SAMPLE_NUM` | Number of validation utterances per held-out book |
| `MAX_WORKERS` | Process pool workers for Arrow packing |

Per-utterance processing details (from the script):

- Speaking-rate label = `count_syllables(text) / duration`, snapped to a discrete class with `map_to_class` (0.25 step). Syllables are counted with `pyphen` for Latin words and one-per-CJK-character for Chinese, plus punctuation counts.
- Train split keeps utterances with `duration >= 2.4s`; validation keeps `2.4s <= duration <= 10s`.
- Books `EN_B00113` and `ZH_B00091` are held out as the validation set; everything else is train.
- Invalid-character utterances, hard-coded bad IDs (`out_zh` / `out_en`), filtered characters, and repetitions are dropped.

The script writes to `data/Emilia_ZH_EN_speed/`:

```text
data/Emilia_ZH_EN_speed/
├── raw.arrow          # train split
├── raw_val.arrow      # validation split
├── duration.json      # train durations
└── duration_val.json  # validation durations
```

The output folder name is `data/{DATASET_NAME}` where `DATASET_NAME = "Emilia_ZH_EN_speed"`. The training loader (`rate_pred.model.dataset.load_dataset`) resolves `datasets.name` by appending the `_speed` suffix, so a config `datasets.name: Emilia_ZH_EN` maps to `data/Emilia_ZH_EN_speed/`.

## Configure Training

The default config is:

```text
src/rate_pred/configs/SpeedPredict_Base_GCE_p3.yaml
```

Important fields:

| Field | Value | Meaning |
| --- | --- | --- |
| `datasets.name` | `Emilia_ZH_EN` | SRP dataset name; loader reads `data/${name}_speed/` |
| `datasets.batch_size_per_gpu` | `19200` | frame budget per GPU (`batch_size_type: frame`) |
| `datasets.max_samples` | `64` | max sequences per frame-batch |
| `model.silence_prob` | `0.3` | probability of applying silence-aware augmentation (read by `train.py`) |
| `model.name` | `SpeedPredictor` | model name (used in the run/checkpoint dir) |
| `model.loss` | `GCE` | Gaussian Cross-Entropy (soft Gaussian labels); `GCE` or `CE` |
| `model.gce_sigma` | `1` | Gaussian soft-label sigma for GCE |
| `model.arch` | `dim=512, depth=6, heads=8, ff_mult=4` | transformer architecture |
| `model.mel_spec` | 24 kHz, 100 mel, vocos | must match CLF52 inference mel config |
| `optim.learning_rate` | `2.5e-4` | peak LR |
| `optim.num_warmup_updates` | `7500` | linear warmup updates |
| `ckpts.logger` | `wandb` | `wandb`, `tensorboard`, or `null` |
| `ckpts.save_dir` | `ckpts/${model.name}_${datasets.name}_${model.loss}_p3` | checkpoint output directory |

Silence-aware augmentation is controlled by three `model` keys that `train.py` reads — `silence_prob` (probability, `0.3`), `silence_ratio_min` (`0.2`), and `silence_ratio_max` (`0.8`). When triggered, `SpeedPredictor._inject_silence_mel_batch` inserts silence at the beginning, end, or both sides of the prompt, with the inserted duration sampled between `silence_ratio_min` and `silence_ratio_max` of the original prompt length. Set `model.silence_prob: 0.0` to disable it.

If your prepared dataset name differs, override it when launching:

```bash
++datasets.name=your_dataset_name
```

## Training

The SRP trainer uses 🤗 Accelerate. Set it up first:

```bash
accelerate config
```

Launch training (run from the repo root):

```bash
accelerate launch src/rate_pred/train/train.py \
  --config-name SpeedPredict_Base_GCE_p3
```

Override config values if needed:

```bash
accelerate launch --mixed_precision=fp16 src/rate_pred/train/train.py \
  --config-name SpeedPredict_Base_GCE_p3 \
  ++datasets.name=Emilia_ZH_EN \
  ++datasets.batch_size_per_gpu=19200 \
  ++model.silence_prob=0.0
```

The training entry point is `rate_pred.train.train:main` (Hydra + Accelerate). The run/experiment name is derived as `${model.name}_${datasets.name}_${model.loss}_p3` (for the default config: `SpeedPredictor_Emilia_ZH_EN_GCE_p3`).

Checkpoints are saved under:

```text
ckpts/${model.name}_${datasets.name}_${model.loss}_p3/...
```

Checkpointing cadence is controlled by `ckpts.save_per_updates`, `ckpts.last_per_updates`, and `ckpts.keep_last_n_checkpoints` in the config.

> Paper training recipe: 6 layers, 8 heads, 512-dim embeddings; GCE loss with Gaussian soft labels (`sigma = 1.0`); balanced 500h English + 500h Chinese Emilia subset; 4 A100 GPUs; 30K updates; per-GPU batch 19,200 frames; LR warmed to 2.5e-4 over the first 7.5K updates then linearly decayed.

## W&B Logging

Logging depends on `ckpts.logger` in the config (default `wandb`; the W&B project is `SpeedPredictor`). A `wandb/` directory is created under the path where you launch training.

Log in to W&B:

```bash
wandb login
```

Or set an API key:

```bash
export WANDB_API_KEY=<YOUR_WANDB_API_KEY>
```

For offline logging:

```bash
export WANDB_MODE=offline
```

To disable logging, set `ckpts.logger=null` (or `++ckpts.logger=tensorboard` to use TensorBoard instead).
