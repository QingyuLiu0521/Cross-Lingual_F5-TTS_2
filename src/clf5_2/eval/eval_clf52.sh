#!/usr/bin/env bash
# Cross-Lingual F5-TTS 2 (CLF5-2) — transcript-free + SRP batch evaluation driver.
#
# Pipeline per (NAME, CKPT, TEST):
#   1) accelerate launch eval_infer_batch.py   -> generate audio (transcript-free, SRP-timed)
#   2) WER + SIM   (seedtts -> eval_seedtts_testset.py ; librispeech -> eval_librispeech_test_clean.py)
#   3) UTMOS       (eval_utmos.py)
#
# Data expected under <repo>/data/ (resolved by the py scripts via files("clf5_2")/../..):
#   - data/librispeech_pc_test_clean_cross_sentence.lst   (for ls_pc_test_clean)
#   - data/seedtts_testset/{zh,en}/meta.lst               (for seedtts_test_{zh,en})
# LibriSpeech audio dir is passed explicitly via LIBRI_PATH below.
set -e

# --------------------------------------------------
# Parameters (edit as needed)
# --------------------------------------------------
# Model config name(s) under clf5_2/configs/ (the final CLF5-2 recipe)
NAMES=(
  "CLF52_Base"
)

# Model checkpoint step(s). The infer script looks for
#   <repo>/ckpts/${NAME}/model_${CKPT}.{pt,safetensors}
# then falls back to <repo>/${model_cfg.ckpts.save_dir}/model_${CKPT}.{pt,safetensors}
CKPTS=(100000)

# Test scenarios
TESTS=(
  "ls_pc_test_clean"
  "seedtts_test_en"
  "seedtts_test_zh"
)

# Inference hyper-params
NFE=32
SEED=0
SCHEDULER="euler"
SS="-1.0"   # must match the float rendering of sway_sampling_coef in the infer output_dir
CFG="2.0"
SPEED="1.0"

# Speaking Rate Predictor (SRP): config under rate_pred/configs/ + its checkpoint step.
# The infer script looks for <repo>/ckpts/${NS}/model_${CS}.pt
NS="SpeedPredict_Base_GCE_p3"
CS=32000

# GPUs used by the metric (WER/SIM) stage
GPU_NUMS=8

# LibriSpeech test-clean audio root (only used by ls_pc_test_clean)
LIBRI_PATH="${LIBRI_PATH:-/path/to/LibriSpeech/test-clean}"

# --------------------------------------------------
EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$EVAL_DIR/../../.." && pwd)"
EVAL_INFER="$EVAL_DIR/eval_infer_batch.py"
EVAL_SEEDTTS="$EVAL_DIR/eval_seedtts_testset.py"
EVAL_LIBRI="$EVAL_DIR/eval_librispeech_test_clean.py"
EVAL_UTMOS="$EVAL_DIR/eval_utmos.py"

for NAME in "${NAMES[@]}"; do
  for C in "${CKPTS[@]}"; do
    for T in "${TESTS[@]}"; do
      echo "===== NAME=$NAME | ckpt=$C | test=$T ====="

      # 1) inference (transcript-free + SRP)
      accelerate launch "$EVAL_INFER" \
        -s 0 -n "$NAME" -c "$C" -t "$T" -nfe "$NFE" -ss "$SS" \
        -ns "$NS" -cs "$CS" --local

      # 2) output dir (must match the path built inside eval_infer_batch.py)
      OUTDIR="$REPO_ROOT/results/${NAME}_${C}/${T}/seed${SEED}_${SCHEDULER}_nfe${NFE}_vocos_ss${SS}_cfg${CFG}_speed${SPEED}_${NS}_${CS}"

      # 3) WER + SIM
      if [[ "$T" == "seedtts_test_zh" || "$T" == "seedtts_test_en" ]]; then
        LANG_CODE="${T##*_}"
        echo " -> eval WER on $OUTDIR"
        python "$EVAL_SEEDTTS" -e wer -l "$LANG_CODE" --gen_wav_dir "$OUTDIR" --local --gpu_nums "$GPU_NUMS"
        echo " -> eval SIM on $OUTDIR"
        python "$EVAL_SEEDTTS" -e sim -l "$LANG_CODE" --gen_wav_dir "$OUTDIR" --local --gpu_nums "$GPU_NUMS"
      else
        echo " -> eval WER on $OUTDIR"
        python "$EVAL_LIBRI" -e wer -l en -g "$OUTDIR" -p "$LIBRI_PATH" --local --gpu_nums "$GPU_NUMS"
        echo " -> eval SIM on $OUTDIR"
        python "$EVAL_LIBRI" -e sim -l en -g "$OUTDIR" -p "$LIBRI_PATH" --local --gpu_nums "$GPU_NUMS"
      fi

      # 4) UTMOS
      echo " -> eval UTMOS on $OUTDIR"
      python "$EVAL_UTMOS" --audio_dir "$OUTDIR"

      echo
    done
  done
done
