# Evaluation

Objective evaluation for **Cross-Lingual F5-TTS 2 (CLF5-2)**: transcript-free zero-shot
voice cloning scored with **WER/CER** (intelligibility), **SIM-o** (speaker similarity),
and **UTMOS** (naturalness) on the standard intra-lingual test sets.

Because CLF5-2 is transcript-free, batch inference **drops the prompt transcript** and lets the
Speaking Rate Predictor (SRP) estimate target duration — the reference text in each `.lst`
is only used as ground truth for WER/CER, never fed to the model.

## Preparation

Install the evaluation extras (adds `faster_whisper`, `funasr`, `jiwer`, `modelscope`,
`zhconv`, `zhon`):

```bash
pip install -e ".[eval]"
```

> [!IMPORTANT]
> For [faster-whisper](https://github.com/SYSTRAN/faster-whisper), for various compatibilities:
> `pip install ctranslate2==4.5.0` if CUDA 12 and cuDNN 9;
> `pip install ctranslate2==4.4.0` if CUDA 12 and cuDNN 8;
> `pip install ctranslate2==3.24.0` if CUDA 11 and cuDNN 8.

### Prepare CLF5-2 checkpoints

Download from the Hugging Face repo
[QingyuLiu1/Cross-Lingual_F5-TTS_2](https://huggingface.co/QingyuLiu1/Cross-Lingual_F5-TTS_2)
and place under `ckpts/` (see `ckpts/README.md`):

| File | Role |
| --- | --- |
| `ckpts/clf52_100000.safetensors` | Main CLF5-2 (synthetic-prompt SFT) model. |
| `ckpts/silence-robust-srp.safetensors` | Silence-robust Speaking Rate Predictor. |
| `ckpts/vocos-mel-24khz/` | Local Vocos mel vocoder (24 kHz, 100 mel). |

> [!NOTE]
> The driver `eval_clf52.sh` resolves checkpoints **by experiment name and step**, i.e.
> `ckpts/<NAME>/model_<CKPT>.{safetensors,pt}` for the TTS model and
> `ckpts/<SRP_NAME>/model_<SRP_CKPT>.pt` for the SRP, falling back to each config's
> `ckpts.save_dir`. Arrange your training/released checkpoints accordingly, or edit the
> `NAMES` / `CKPTS` / `NS` / `CS` variables at the top of the script.

### Prepare test datasets

Place the following under `data/` at the repo root (paths are resolved by the Python scripts
relative to the package via `importlib.resources.files("clf5_2")/../../`):

1. **Seed-TTS test-zh / test-en**: download from
   [seed-tts-eval](https://github.com/BytedanceSpeech/seed-tts-eval) and provide
   `data/seedtts_testset/zh/meta.lst` and `data/seedtts_testset/en/meta.lst`.
2. **LibriSpeech-PC test-clean**: provide the cross-sentence list
   `data/librispeech_pc_test_clean_cross_sentence.lst`; the raw `test-clean` audio
   (`.flac`) lives in a separate directory that you pass explicitly (see `LIBRI_PATH` below).

### Prepare evaluation models

| Metric | Model | Notes |
| --- | --- | --- |
| WER (EN) | [faster-whisper-large-v3](https://huggingface.co/Systran/faster-whisper-large-v3) | Auto-downloaded to the cache dir unless `--local` is set. |
| CER (ZH) | [Paraformer-zh](https://huggingface.co/funasr/paraformer-zh) (funasr) | Auto-downloaded to the cache dir unless `--local` is set. |
| SIM-o | WavLM-large speaker-verification checkpoint (`wavlm_large_finetune.pth`) | Loaded by the `ECAPA_TDNN_SMALL` wrapper in `ecapa_tdnn.py`. |
| UTMOS | `tarepan/SpeechMOS:v1.2.0 → utmos22_strong` | Fetched via `torch.hub`. |

> [!NOTE]
> ASR models auto-download unless you pass `--local`. To use pre-downloaded checkpoints,
> pass `--local` and update the `asr_ckpt_dir` paths in
> [`eval_seedtts_testset.py`](eval_seedtts_testset.py) /
> [`eval_librispeech_test_clean.py`](eval_librispeech_test_clean.py).
> The WavLM path is set by `wavlm_ckpt_dir` in those same scripts and must point at your
> downloaded `wavlm_large_finetune.pth`.

## Run the full pipeline

The one-shot driver runs, for every `(NAME, CKPT, TEST)` combination:
**(1)** transcript-free + SRP-timed batch inference, **(2)** WER/CER + SIM scoring, and
**(3)** UTMOS.

```bash
LIBRI_PATH=/path/to/LibriSpeech/test-clean bash src/clf5_2/eval/eval_clf52.sh
```

Edit the parameter block at the top of
[`eval_clf52.sh`](eval_clf52.sh) to control the run:

| Variable | Description |
| --- | --- |
| `NAMES` | Model config name(s) under `src/clf5_2/configs/` (default `CLF52_Base`). |
| `CKPTS` | Model checkpoint step(s) to evaluate (default `100000`). |
| `TESTS` | Test scenarios: `ls_pc_test_clean`, `seedtts_test_en`, `seedtts_test_zh`. |
| `NFE` | Number of function evaluations for the ODE sampler (default `32`). |
| `SEED` | Sampling seed (default `0`). |
| `SCHEDULER` | ODE method (default `euler`). |
| `SS` | Sway-sampling coefficient (default `-1.0`). |
| `CFG` | Classifier-free guidance strength (default `2.0`). |
| `SPEED` | Speed factor (default `1.0`). |
| `NS` / `CS` | SRP config name (under `src/rate_pred/configs/`) and its checkpoint step (default `SpeedPredict_Base_GCE_p3` / `32000`). |
| `GPU_NUMS` | GPUs used by the WER/SIM scoring stage (default `8`). |
| `LIBRI_PATH` | LibriSpeech `test-clean` audio root; only used by `ls_pc_test_clean`. |

Generated audio is written to (and metric `.jsonl` files are dropped alongside):

```text
results/<NAME>_<CKPT>/<TEST>/seed<SEED>_<SCHEDULER>_nfe<NFE>_vocos_ss<SS>_cfg<CFG>_speed<SPEED>_<NS>_<CS>/
```

> [!IMPORTANT]
> `SS` must match the float rendering of the sway-sampling coefficient in the output path
> (e.g. `-1.0`), otherwise the scoring stage will not find the generated audio.

## Scripts

| Script | Purpose |
| --- | --- |
| [`eval_clf52.sh`](eval_clf52.sh) | Driver: chains batch inference → WER/SIM → UTMOS for each `(NAME, CKPT, TEST)`. |
| [`eval_infer_batch.py`](eval_infer_batch.py) | Transcript-free, SRP-timed batch synthesis with `CFM_SFT`. |
| [`eval_seedtts_testset.py`](eval_seedtts_testset.py) | WER/CER + SIM on Seed-TTS test-zh / test-en. |
| [`eval_librispeech_test_clean.py`](eval_librispeech_test_clean.py) | WER + SIM on LibriSpeech-PC test-clean. |
| [`eval_utmos.py`](eval_utmos.py) | UTMOS naturalness over a directory of `.wav`. |
| [`ecapa_tdnn.py`](ecapa_tdnn.py) | WavLM-large speaker-embedding model used for SIM (third-party). |
| [`utils_eval.py`](utils_eval.py) | Shared helpers: metainfo readers, prompt construction, ASR/SIM runners. |

### Batch inference (transcript-free + SRP)

Runs under `accelerate` (DDP, one utterance per process). The prompt transcript is dropped
(`droptext=True`); the SRP predicts target duration from the reference mel, and the missing
prompt text is replaced inside the model by ratio-based learnable prompt tokens `<P>` plus the
EOS anchor `". "`. Audio is decoded with the local Vocos vocoder.

```bash
accelerate launch src/clf5_2/eval/eval_infer_batch.py \
  -s 0 -n CLF52_Base -c 100000 -t seedtts_test_en -nfe 32 -ss -1.0 \
  -ns SpeedPredict_Base_GCE_p3 -cs 32000 --local
```

| Argument | Description |
| --- | --- |
| `-s`, `--seed` | Sampling seed. |
| `-n`, `--expname` | TTS model config name under `src/clf5_2/configs/` (**required**). |
| `-c`, `--ckptstep` | TTS checkpoint step (default `1250000`). |
| `-ns`, `--expnamesp` | SRP config name under `src/rate_pred/configs/` (**required**). |
| `-cs`, `--ckptstepsp` | SRP checkpoint step (default `10000`). |
| `-nfe`, `--nfestep` | ODE function evaluations (default `32`). |
| `-o`, `--odemethod` | ODE method (default `euler`). |
| `-ss`, `--swaysampling` | Sway-sampling coefficient (default `-1`). |
| `-t`, `--testset` | Test set: `ls_pc_test_clean`, `seedtts_test_en`, `seedtts_test_zh`, `cross_lingual`, … (**required**). |
| `-p`, `--librispeech_test_clean_path` | LibriSpeech `test-clean` audio root (used by `ls_pc_test_clean`). |
| `--cross_lingual_subset` | Cross-lingual pair (e.g. `ko2en`, `fr2zh`, `mt2en`); required when `-t cross_lingual`. |
| `--cross_lingual_lst_dir` / `--cross_lingual_wav_dir` | List dir / wav dir for cross-lingual eval; required when `-t cross_lingual`. |
| `--local` | Load the Vocos vocoder from the local `ckpts/vocos-mel-24khz` dir. |

> [!NOTE]
> Cross-lingual subsets follow the paper's 16 prompt→target pairs — each of the 8
> out-of-distribution prompt languages (`ko`, `th`, `vi`, `id`, `fr`, `de`, `ru`, `mt`)
> paired with English (`2en`) and Chinese (`2zh`) targets.

### Metric scoring

Both scorers split the generated set across GPUs and pool per-rank results. WER (EN) uses
faster-whisper-large-v3, CER (ZH) uses funasr Paraformer-zh, and SIM computes the cosine
similarity between generated and prompt speaker embeddings. Per-sample results and the mean
are written to `_<task>_results.jsonl` inside the generated-audio directory.

**Seed-TTS test-zh / test-en:**

```bash
# WER/CER
python src/clf5_2/eval/eval_seedtts_testset.py -e wer -l en \
  --gen_wav_dir results/CLF52_Base_100000/seedtts_test_en/<run-dir> --local --gpu_nums 8
# SIM
python src/clf5_2/eval/eval_seedtts_testset.py -e sim -l en \
  --gen_wav_dir results/CLF52_Base_100000/seedtts_test_en/<run-dir> --local --gpu_nums 8
```

| Argument | Description |
| --- | --- |
| `-e`, `--eval_task` | `wer` or `sim`. |
| `-l`, `--lang` | `en` or `zh`. |
| `-g`, `--gen_wav_dir` | Directory of generated `.wav` to score (**required**). |
| `--metalst` | Optional override for the Seed-TTS `meta.lst`. |
| `-n`, `--gpu_nums` | GPU count (e.g. `8`) or list (e.g. `[0,1,2,3]`). |
| `--local` | Use the local ASR checkpoint dirs instead of auto-download. |

**LibriSpeech-PC test-clean:**

```bash
# WER
python src/clf5_2/eval/eval_librispeech_test_clean.py -e wer -l en \
  -g results/CLF52_Base_100000/ls_pc_test_clean/<run-dir> \
  -p /path/to/LibriSpeech/test-clean --local --gpu_nums 8
# SIM
python src/clf5_2/eval/eval_librispeech_test_clean.py -e sim -l en \
  -g results/CLF52_Base_100000/ls_pc_test_clean/<run-dir> \
  -p /path/to/LibriSpeech/test-clean --local --gpu_nums 8
```

| Argument | Description |
| --- | --- |
| `-e`, `--eval_task` | `wer` or `sim`. |
| `-l`, `--lang` | Language (default `en`). |
| `-g`, `--gen_wav_dir` | Directory of generated `.wav` to score (**required**). |
| `-p`, `--librispeech_test_clean_path` | LibriSpeech `test-clean` audio root (**required**). |
| `-n`, `--gpu_nums` | GPU count (e.g. `8`) or list (e.g. `[0,1,2,3]`). |
| `--local` | Use the local ASR checkpoint dir instead of auto-download. |

### UTMOS

Scores every `*.wav` under a directory with `utmos22_strong` and writes
`_utmos_results.jsonl` (per-sample scores + mean).

```bash
python src/clf5_2/eval/eval_utmos.py \
  --audio_dir results/CLF52_Base_100000/seedtts_test_en/<run-dir>
```

| Argument | Description |
| --- | --- |
| `--audio_dir` | Directory to score, searched recursively (**required**). |
| `--ext` | Audio extension (default `wav`). |

## Metrics

- **WER / CER** — Whisper-large-v3 (English) / Paraformer-zh (Chinese) transcription against
  the reference text, punctuation-stripped; Chinese is scored character-level.
- **SIM-o** — cosine similarity between the WavLM-large speaker embeddings of the generated
  and prompt audio.
- **UTMOS** — objective naturalness proxy (`utmos22_strong`).

On the intra-lingual test sets, Cross-Lingual F5-TTS 2 reaches the highest SIM-o of the
compared systems (LibriSpeech-PC test-clean 0.687, Seed-TTS test-en 0.683, test-zh 0.768)
while keeping WER/CER competitive (lowest WER on LibriSpeech-PC test-clean at 2.014%), at a
small cost in UTMOS. See the paper for the complete tables.
