# Cross-Lingual F5-TTS 2 Inference

This folder contains the transcript-free command-line inference entry point for Cross-Lingual F5-TTS 2 (CLF5-2):

- `infer_cli.py`: transcript-free zero-shot voice cloning. Requires only reference audio; the reference transcript is **not** needed and is ignored. Duration is predicted by the silence-robust speaking rate predictor (SRP), and the missing prompt transcript is replaced inside the model by ratio-based learnable prompt tokens plus an EOS anchor.

Unlike vanilla F5-TTS, which needs the transcript of the audio prompt at inference, CLF5-2 is a **single-model**, transcript-free system: one checkpoint clones any reference voice, in any language, without a transcript.

All commands below should be run from the repository root (`Cross-Lingual_F5-TTS_2/`).

Pretrained CLF5-2 checkpoints are available on [Hugging Face](https://huggingface.co/QingyuLiu1/Cross-Lingual_F5-TTS_2). Download the needed files, place them under `ckpts/`, and set the paths in the TOML file or pass them with command-line flags. See `ckpts/README.md` for details.

## CLF5-2 CLI

The package installs a console script `clf52-infer` (declared in `pyproject.toml` as `clf52-infer = "clf5_2.infer.infer_cli:main"`). Run it with the packaged example TOML:

```bash
clf52-infer -c src/clf5_2/infer/examples/clf52/clf52.toml
```

The equivalent module form is:

```bash
python -m clf5_2.infer.infer_cli -c src/clf5_2/infer/examples/clf52/clf52.toml
```

To pin a GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python -m clf5_2.infer.infer_cli -c src/clf5_2/infer/examples/clf52/clf52.toml
```

The matching config is:

```text
src/clf5_2/infer/examples/clf52/clf52.toml
```

## How CLF5-2 Inference Works

At inference time CLF5-2 differs from a standard F5-TTS pipeline in three ways:

1. **Transcript-free.** The reference transcript (`ref_text` / `-s`) is ignored. You never provide, transcribe, or align the prompt audio.
2. **SRP-predicted duration.** The silence-robust SRP predicts a syllable-level speaking rate from the prompt mel-spectrogram. The target duration is computed as `target linguistic units / predicted speaking rate` (with the `--speed` and `--duration_scale` multipliers applied on top), so no reference-text length ratio is needed.
3. **Prompt tokens instead of a prompt transcript.** Inside the model, the dropped prompt transcript is replaced by `N` ratio-based learnable prompt tokens (`<P>`) followed by an EOS anchor (`". "`), then the target text. `N` is set from the prompt-to-target duration ratio. This is enabled automatically through the model config keys `prefix_text_pad`, `prompt_token`, and `eos_marker`; when `prompt_token` is set, the backbone vocabulary is extended by one learnable token id.

Long inputs are split into chunks bounded by the SRP-predicted speaking rate and cross-fade concatenated, so long text is not compressed into a single fixed-length generation.

## `clf52.toml` Fields

Every field of the packaged `src/clf5_2/infer/examples/clf52/clf52.toml`:

| Field | Meaning | Example value |
| --- | --- | --- |
| `ckpt_file` | CLF5-2 model checkpoint (`.safetensors`/`.pt`) | `ckpts/clf52_100000.safetensors` |
| `srp_ckpt_file` | Silence-robust SRP checkpoint. **Required** — inference aborts if unset | `ckpts/silence-robust-srp.safetensors` |
| `vocoder_name` | Vocoder type: `vocos` or `bigvgan` | `vocos` |
| `load_vocoder_from_local` | Load the vocoder from a local dir (`ckpts/vocos-mel-24khz` for Vocos) instead of downloading | `true` |
| `ref_audio` | Reference voice audio (any language; no transcript needed). Paths under `infer/examples/` resolve against the packaged assets | `infer/examples/basic/basic_ref_en.wav` |
| `ref_text` | **Ignored** by CLF5-2; kept only for config/CLI compatibility | `""` |
| `gen_text` | Text to synthesize (may differ in language from the prompt) | `"This is a quick test of Cross-Lingual F five T T S two."` |
| `gen_file` | Path to a UTF-8 text file to synthesize; if set, overrides `gen_text` | `""` |
| `remove_silence` | Remove long silence from the final wav | `false` |
| `output_dir` | Output folder (relative to the current working directory) | `tests` |
| `output_file` | Output filename | `clf52_basic.wav` |

The model config and SRP config default to the packaged copies (`clf5_2/configs/CLF52_Base.yaml` and `rate_pred/configs/SpeedPredict_Base_GCE_p3.yaml`), so they are omitted from the example TOML. Override them from the config file (`model_cfg` / `srp_model_cfg`) or from the command line if needed.

## Command-Line Flags

All TOML fields can be overridden on the command line. CLI flags take precedence over the config file.

| Flag | Meaning |
| --- | --- |
| `-c`, `--config` | Config TOML (default: packaged `infer/examples/clf52/clf52.toml`) |
| `-m`, `--model` | Model config **name** under `clf5_2/configs` (falls back to `CLF52_Base` when unset in both CLI and TOML) |
| `-mc`, `--model_cfg` | Path to the model config `.yaml`, overrides `-m` |
| `-p`, `--ckpt_file` | CLF5-2 model checkpoint path |
| `-v`, `--vocab_file` | Vocab `.txt`; leave blank to use the dataset/tokenizer default |
| `--srp_ckpt_file` | SRP checkpoint path (**required**) |
| `--srp_model_cfg` | SRP config `.yaml` (default `rate_pred/configs/SpeedPredict_Base_GCE_p3.yaml`) |
| `-r`, `--ref_audio` | Reference audio file |
| `-s`, `--ref_text` | **Ignored** (transcript-free); present for compatibility |
| `-t`, `--gen_text` | Text to synthesize |
| `-f`, `--gen_file` | Text file to synthesize (overrides `--gen_text`) |
| `-o`, `--output_dir` | Output folder |
| `-w`, `--output_file` | Output filename |
| `--save_chunk` | Save each generated chunk separately |
| `--no_legacy_text` | Do **not** use lossy ASCII transliterations for chunk file names |
| `--remove_silence` | Remove long silence from the output |
| `--load_vocoder_from_local` | Load vocoder from a local dir (`ckpts/vocos-mel-24khz`) |
| `--vocoder_name` | `vocos` or `bigvgan` (default `vocos`) |
| `--target_rms` | Output loudness normalization target (default `0.1`) |
| `--cross_fade_duration` | Cross-fade seconds between chunks (default `0.15`) |
| `--nfe_step` | Denoising / ODE steps (default `32`) |
| `--cfg_strength` | Classifier-free guidance strength (default `2.0`) |
| `--sway_sampling_coef` | Sway sampling coefficient (default `-1.0`) |
| `--speed` | Speed multiplier applied on top of the SRP-predicted duration (default `1.0`) |
| `--duration_scale` | Multiply the SRP-predicted duration by this factor, e.g. `0.85` to shorten (default `1.0`) |
| `--chunk_silence` | Seconds of silence inserted between consecutive chunks of split long text (default `0.0`) |
| `--seed` | Random seed for reproducible sampling |
| `--device` | Device to run on (auto-detects CUDA / XPU / MPS / CPU) |

Example overriding checkpoints and target text directly:

```bash
python -m clf5_2.infer.infer_cli \
  -c src/clf5_2/infer/examples/clf52/clf52.toml \
  --ckpt_file ckpts/clf52_100000.safetensors \
  --srp_ckpt_file ckpts/silence-robust-srp.safetensors \
  --gen_text "Synthesize this line in the reference voice." \
  --output_dir tests \
  --output_file my_clone.wav
```

Example overriding the model / SRP config paths:

```bash
python -m clf5_2.infer.infer_cli \
  -c src/clf5_2/infer/examples/clf52/clf52.toml \
  --model_cfg src/clf5_2/configs/CLF52_Base.yaml \
  --srp_model_cfg src/rate_pred/configs/SpeedPredict_Base_GCE_p3.yaml
```

## Multi-Voice TOML

The CLI supports multiple voices through `[voices.<name>]`. Because CLF5-2 is transcript-free, each voice entry only needs `ref_audio` and an optional per-voice `speed`. Voice segments are selected in `gen_text` with `[name]` tags; text without a tag uses `main`.

```toml
ref_audio = "infer/examples/basic/basic_ref_en.wav"
ref_text = ""  # ignored

gen_text = """
[main]This line uses the main reference voice.
[alice]And this line uses Alice's voice.
"""

[voices.alice]
ref_audio = "path/to/alice.wav"
speed = 1.0
```

## Packaged Example Assets

The following assets ship inside the `clf5_2` package under `src/clf5_2/infer/examples/`, so the example config runs out of the box after `pip install -e .`:

```text
src/clf5_2/infer/examples/
├── vocab.txt                    # tokenizer vocabulary
├── basic/
│   └── basic_ref_en.wav         # default English reference prompt
└── clf52/
    └── clf52.toml               # default inference config
```

Any `ref_audio` / `gen_file` path containing `infer/examples/` is resolved against the packaged copy automatically when a local file of that name does not exist, so the default TOML works from any working directory.

## Checkpoints

Download from the Hugging Face repo `QingyuLiu1/Cross-Lingual_F5-TTS_2` and place the files under `ckpts/`:

```text
ckpts/clf52_100000.safetensors          # main CLF5-2 (synthetic-prompt SFT) model
ckpts/silence-robust-srp.safetensors    # silence-robust SRP model
ckpts/vocos-mel-24khz/                   # local Vocos mel vocoder (24 kHz, 100 mel)
```

The example TOML references these paths directly. See `ckpts/README.md` for the full download instructions.
