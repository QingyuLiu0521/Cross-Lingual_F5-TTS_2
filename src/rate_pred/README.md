# Speaking Rate Predictor

Speaking Rate Predictor (SRP) is the duration helper used by Cross-Lingual F5-TTS 2, following Cross-Lingual F5-TTS (Proc. ICASSP 2026).

CLF5-2 is transcript-free: it does not use the audio prompt's transcript at inference. Because it cannot estimate the generation length from a reference text/audio ratio, it uses the SRP to predict the prompt's speaking rate from the reference audio, then estimates the target duration from the generated text.

The SRP here adds one thing over the original Cross-Lingual F5-TTS predictor: **silence-aware augmentation**, which makes the transcript-free duration estimate robust to leading/trailing silence in the audio prompt. Without it, silent regions deflate the estimated speaking rate and inflate the synthesized duration.

## What SRP Does

SRP takes reference audio (raw wave or mel) and predicts a syllable-level speaking-rate bucket. During CLF5-2 inference, this predicted speed is used to:

- estimate the duration of the generated audio: `predicted duration = target syllable count / predicted speaking rate`
- keep transcript-free inference stable without requiring the prompt transcript

The predictor is a small transformer (`SpeedTransformer`): a mel projection, a `ConvPositionEmbedding`, rotary attention blocks, attention pooling over time, and a classifier head over `num_classes = 32` discrete speaking-rate buckets. Buckets are mapped back to a scalar speaking rate by `SpeedMapper` (`speed = label * 0.25 + 0.25`). It is trained with Gaussian Cross-Entropy (GCE) using Gaussian soft labels.

## Files

```text
src/rate_pred/
├── configs/
│   └── SpeedPredict_Base_GCE_p3.yaml
├── model/
│   ├── __init__.py
│   ├── speed_predictor.py
│   ├── dataset.py
│   ├── trainer.py
│   ├── modules.py
│   └── utils.py
└── train/
    ├── train.py
    └── datasets/
        └── prepare_emilia_speed.py
```

Key files:

| File | Purpose |
| --- | --- |
| `configs/SpeedPredict_Base_GCE_p3.yaml` | Default SRP training config |
| `model/speed_predictor.py` | `SpeedPredictor` model, `predict_speed()`, `SpeedMapper`, and `_inject_silence_mel_batch` |
| `model/dataset.py` | Loads prepared SRP Arrow data (`CustomDataset`, `load_dataset`, `collate_fn_sp`) |
| `model/trainer.py` | `Trainer` used by the training entry point |
| `model/modules.py` | Mel spectrogram, positional/convolution modules, `SpeedPredictorLayer`, `GaussianCrossEntropyLoss` |
| `model/utils.py` | Syllable-count and tensor helpers (`count_syllables`, `lens_to_mask`) |
| `train/datasets/prepare_emilia_speed.py` | Builds SRP training data from the Emilia dataset |
| `train/train.py` | SRP training entry point |

## Silence-Aware Augmentation

The augmentation lives in `SpeedPredictor._inject_silence_mel_batch` and is applied only during training, gated by `silence_prob`:

- **Batch-level gating.** With probability `1 - silence_prob` the batch is left unchanged; otherwise the whole batch is augmented.
- **Silence budget.** A silence ratio is uniformly sampled in `[silence_ratio_min, silence_ratio_max]` and multiplied by the batch's longest valid length to get a total silent frame budget `sil_len`.
- **Placement.** For each sample the budget is split randomly between the front and the back (either side can be zero), and zero-valued (silent) mel frames are concatenated before/after the valid frames. Lengths are updated to include the added silence.

The default training config sets `silence_prob = 0.3` (under `model`), and the released silence-robust checkpoint was trained with this augmentation. The augmentation only changes the acoustic input; speaking-rate labels and the model architecture are unchanged. The result is a predictor whose duration estimate stays accurate on prompts that contain leading/trailing silence, at essentially no cost on clean prompts.

`SpeedPredictor` exposes the augmentation knobs in its constructor:

| Argument | Default | Meaning |
| --- | --- | --- |
| `silence_prob` | `0.0` | Probability the batch is augmented (constructor default; the training config sets `0.3` under `model`) |
| `silence_ratio_min` | `0.2` | Lower bound of the silence-to-prompt ratio |
| `silence_ratio_max` | `0.8` | Upper bound of the silence-to-prompt ratio |

## Dataset

The SRP is trained on a balanced English + Chinese subset of the [Emilia](https://huggingface.co/datasets/amphion/Emilia-Dataset) dataset. The preprocessing script reads per-language `*.jsonl` shards, computes a syllable-level speaking rate for each utterance (`count_syllables(text) / duration`, quantized to a `0.25`-wide bucket), filters dirty/repetitive/out-of-charset samples, and packs the result into Arrow files.

Configure paths inside `train/datasets/prepare_emilia_speed.py` (the `__main__` block) before running:

| Constant | Meaning |
| --- | --- |
| `DATASET_DIR` | Root of the local Emilia data (per-language `ZH/`, `EN/` folders of `*.jsonl`) |
| `LANGS` | Languages to include (default `["ZH", "EN"]`) |
| `MAX_DURATION_HOURS` | Per-language training hours (default `500`, so `500 + 500 = 1000h`) |
| `VAL_SAMPLE_NUM` | Validation samples per language |

Prepare SRP data:

```bash
python src/rate_pred/train/datasets/prepare_emilia_speed.py
```

This writes to `data/Emilia_ZH_EN_speed/` (the script sets `DATASET_NAME = f"Emilia_{'_'.join(LANGS)}_speed"`):

```text
data/Emilia_ZH_EN_speed/
├── raw.arrow
├── raw_val.arrow
├── duration.json
└── duration_val.json
```

`load_dataset` in `model/dataset.py` looks the data up by name under `data/<name>_speed/`, so the config's `datasets.name` (default `Emilia_ZH_EN`) must match the prepared folder.

## Training

The default config is:

```text
src/rate_pred/configs/SpeedPredict_Base_GCE_p3.yaml
```

Important fields:

| Field | Meaning |
| --- | --- |
| `datasets.name` | SRP dataset name, e.g. `Emilia_ZH_EN` (data lives at `data/<name>_speed/`) |
| `datasets.batch_size_per_gpu` | frame budget per GPU (default `19200`, frame-wise) |
| `model.silence_prob` | probability of applying silence-aware augmentation (default `0.3`) |
| `model.loss` | `GCE` or `CE` (default `GCE`) |
| `model.gce_sigma` | Gaussian soft-label sigma for GCE (default `1`) |
| `model.arch` | transformer arch (`dim=512`, `depth=6`, `heads=8`, `ff_mult=4`) |
| `model.mel_spec` | mel configuration (24 kHz, 100 mel, `vocos`); should match CLF5-2 inference |
| `ckpts.save_dir` | checkpoint output directory |
| `ckpts.logger` | `wandb`, `tensorboard`, or `null` |

Set up `accelerate` first:

```bash
accelerate config
```

Launch training with the default config:

```bash
accelerate launch src/rate_pred/train/train.py \
  --config-name SpeedPredict_Base_GCE_p3
```

Override config values if needed (for example a different prepared dataset name or batch budget):

```bash
accelerate launch src/rate_pred/train/train.py \
  --config-name SpeedPredict_Base_GCE_p3 \
  ++datasets.name=your_dataset_name \
  ++datasets.batch_size_per_gpu=19200
```

Checkpoints are saved under:

```text
ckpts/${model.name}_${datasets.name}_${model.loss}_p3/...
```

Logging follows `ckpts.logger` in the config; the default is [Weights & Biases](https://wandb.ai/) (`wandb`). Log in once with `wandb login` (or set `WANDB_API_KEY`) before launching, or set `ckpts.logger=null` to disable it.

## Inference

CLF5-2 inference loads an SRP checkpoint through the main inference CLI:

```text
src/clf5_2/infer/infer_cli.py
```

The CLI (`load_srp_model`) reads the SRP config and checkpoint, builds the `SpeedPredictor`, and calls `predict_speed(audio=ref_mel, lens=ref_mel_len_t)` once on the prompt to obtain the speaking rate used for duration estimation. The SRP checkpoint is **required** for CLF5-2 inference.

Set the checkpoint in the inference TOML (`src/clf5_2/infer/examples/clf52/clf52.toml`):

```toml
srp_ckpt_file = "ckpts/silence-robust-srp.safetensors"
```

Then run inference from the repo root:

```bash
clf52-infer -c src/clf5_2/infer/examples/clf52/clf52.toml
# or: python -m clf5_2.infer.infer_cli -c src/clf5_2/infer/examples/clf52/clf52.toml
```

The SRP config defaults to the packaged copy (`rate_pred/configs/SpeedPredict_Base_GCE_p3.yaml`); override it with `--srp_ckpt_file` / `--srp_model_cfg` if needed. Download `ckpts/silence-robust-srp.safetensors` from the Hugging Face repo [QingyuLiu1/Cross-Lingual_F5-TTS_2](https://huggingface.co/QingyuLiu1/Cross-Lingual_F5-TTS_2) and place it under `ckpts/`.

See the CLF5-2 inference [README](../clf5_2/infer/README.md) for full usage.
