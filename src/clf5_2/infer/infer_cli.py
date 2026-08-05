import argparse
import codecs
import os
import re
from datetime import datetime
from importlib.resources import files
from pathlib import Path

import numpy as np
import soundfile as sf
import tomli
import torch
import torchaudio
from hydra.utils import get_class
from omegaconf import OmegaConf
from unidecode import unidecode

from clf5_2.eval.utils_eval import count  # syllable-level unit counter used by the SRP
from clf5_2.infer.utils_infer import (
    cfg_strength,
    cross_fade_duration,
    device,
    load_checkpoint,
    load_srp_model,
    load_vocoder,
    mel_spec_type,
    nfe_step,
    remove_silence_for_generated_wav,
    speed,
    sway_sampling_coef,
    target_rms,
)
from clf5_2.model import CFM_SFT  # noqa: F401
from clf5_2.model.modules import MelSpec
from clf5_2.model.utils import convert_char_to_pinyin, get_tokenizer


# =============================================================================
# Cross-Lingual F5-TTS 2 (CLF5-2) command-line inference.
#
# Unlike vanilla F5-TTS, CLF5-2 does NOT need the transcript of the audio prompt
# at inference time. The reference text (-s/--ref_text) is therefore ignored.
# Duration is estimated by a syllable-level, silence-robust speaking rate
# predictor (SRP), and the missing prompt transcript is replaced inside the
# model by ratio-based learnable prompt tokens followed by an EOS marker
# (handled automatically via prefix_text_pad / prompt_token /
# eos_marker in the model config).
#
# Reference implementation: eval/eval_infer_batch.py
# =============================================================================


parser = argparse.ArgumentParser(
    prog="python3 infer_cli.py",
    description="Command-line interface for Cross-Lingual F5-TTS 2 (transcript-free) inference.",
    epilog="Specify options above to override one or more settings from config.",
)
parser.add_argument(
    "-c",
    "--config",
    type=str,
    default=os.path.join(files("clf5_2").joinpath("infer/examples/clf52"), "clf52.toml"),
    help="The configuration file, default see infer/examples/clf52/clf52.toml",
)

# Note. Not to provide default value here in order to read default from config file

parser.add_argument(
    "-m",
    "--model",
    type=str,
    help="The model config name under clf5_2/configs (the CLF5-2 CFM_SFT config).",
)
parser.add_argument(
    "-mc",
    "--model_cfg",
    type=str,
    help="The path to the model config .yaml (overrides -m).",
)
parser.add_argument(
    "-p",
    "--ckpt_file",
    type=str,
    help="The path to the CLF5-2 model checkpoint .safetensors/.pt.",
)
parser.add_argument(
    "-v",
    "--vocab_file",
    type=str,
    help="The path to vocab file .txt, leave blank to use dataset default.",
)
parser.add_argument(
    "--srp_ckpt_file",
    type=str,
    help="The path to the speaking rate predictor (SRP) checkpoint. Required.",
)
parser.add_argument(
    "--srp_model_cfg",
    type=str,
    help="The path to the SRP config .yaml (default configs/SpeedPredict_Base_GCE_p3.yaml).",
)
parser.add_argument(
    "-r",
    "--ref_audio",
    type=str,
    help="The reference audio file.",
)
parser.add_argument(
    "-s",
    "--ref_text",
    type=str,
    help="Ignored in CLF5-2 (transcript-free); kept for config/CLI compatibility.",
)
parser.add_argument(
    "-t",
    "--gen_text",
    type=str,
    help="The text to synthesize.",
)
parser.add_argument(
    "-f",
    "--gen_file",
    type=str,
    help="The file with text to generate, will ignore --gen_text.",
)
parser.add_argument(
    "-o",
    "--output_dir",
    type=str,
    help="The path to output folder.",
)
parser.add_argument(
    "-w",
    "--output_file",
    type=str,
    help="The name of output file.",
)
parser.add_argument(
    "--save_chunk",
    action="store_true",
    help="To save each audio chunk during inference.",
)
parser.add_argument(
    "--no_legacy_text",
    action="store_false",
    help="Not to use lossy ASCII transliterations of unicode text in saved file names.",
)
parser.add_argument(
    "--remove_silence",
    action="store_true",
    help="To remove long silence found in output.",
)
parser.add_argument(
    "--load_vocoder_from_local",
    action="store_true",
    help="To load vocoder from local dir, default to ckpts/vocos-mel-24khz.",
)
parser.add_argument(
    "--vocoder_name",
    type=str,
    choices=["vocos", "bigvgan"],
    help=f"Used vocoder name: vocos | bigvgan, default {mel_spec_type}",
)
parser.add_argument(
    "--target_rms",
    type=float,
    help=f"Target output speech loudness normalization value, default {target_rms}",
)
parser.add_argument(
    "--cross_fade_duration",
    type=float,
    help=f"Duration of cross-fade between audio segments in seconds, default {cross_fade_duration}",
)
parser.add_argument(
    "--nfe_step",
    type=int,
    help=f"The number of function evaluation (denoising steps), default {nfe_step}",
)
parser.add_argument(
    "--cfg_strength",
    type=float,
    help=f"Classifier-free guidance strength, default {cfg_strength}",
)
parser.add_argument(
    "--sway_sampling_coef",
    type=float,
    help=f"Sway Sampling coefficient, default {sway_sampling_coef}",
)
parser.add_argument(
    "--speed",
    type=float,
    help=f"Speed multiplier applied on top of the SRP-predicted duration, default {speed}",
)
parser.add_argument(
    "--duration_scale",
    type=float,
    help="Multiply the SRP-predicted target duration by this factor (e.g. 0.85 to shorten), default 1.0",
)
parser.add_argument(
    "--chunk_silence",
    type=float,
    help="Seconds of silence inserted between consecutive chunks when long text is split, default 0.0",
)
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="Random seed for reproducible sampling.",
)
parser.add_argument(
    "--device",
    type=str,
    help="Specify the device to run on.",
)
args = parser.parse_args()


# config file

config = tomli.load(open(args.config, "rb"))


# command-line interface parameters

model = args.model or config.get("model", "CLF52_Base")
ckpt_file = args.ckpt_file or config.get("ckpt_file", "")
vocab_file = args.vocab_file or config.get("vocab_file", "")
srp_ckpt_file = args.srp_ckpt_file or config.get("srp_ckpt_file", "")
srp_model_cfg_file = (
    args.srp_model_cfg
    or config.get("srp_model_cfg")
    or str(files("rate_pred").joinpath("configs/SpeedPredict_Base_GCE_p3.yaml"))
)

ref_audio = args.ref_audio or config.get("ref_audio", "infer/examples/basic/basic_ref_en.wav")
# ref_text is intentionally ignored; CLF5-2 is transcript-free.
gen_text = args.gen_text or config.get("gen_text", "Here we generate something just for test.")
gen_file = args.gen_file or config.get("gen_file", "")

output_dir = args.output_dir or config.get("output_dir", "tests")
output_file = args.output_file or config.get(
    "output_file", f"infer_cli_{datetime.now().strftime(r'%Y%m%d_%H%M%S')}.wav"
)

save_chunk = args.save_chunk or config.get("save_chunk", False)
use_legacy_text = args.no_legacy_text or config.get("no_legacy_text", False)  # no_legacy_text is a store_false arg
if save_chunk and use_legacy_text:
    print(
        "\nWarning to --save_chunk: lossy ASCII transliterations of unicode text for legacy (.wav) file names, --no_legacy_text to disable.\n"
    )

remove_silence = args.remove_silence or config.get("remove_silence", False)
load_vocoder_from_local = args.load_vocoder_from_local or config.get("load_vocoder_from_local", False)

vocoder_name = args.vocoder_name or config.get("vocoder_name", mel_spec_type)
target_rms = args.target_rms or config.get("target_rms", target_rms)
cross_fade_duration = args.cross_fade_duration or config.get("cross_fade_duration", cross_fade_duration)
nfe_step = args.nfe_step or config.get("nfe_step", nfe_step)
cfg_strength = args.cfg_strength or config.get("cfg_strength", cfg_strength)
sway_sampling_coef = args.sway_sampling_coef or config.get("sway_sampling_coef", sway_sampling_coef)
speed = args.speed or config.get("speed", speed)
duration_scale = args.duration_scale or config.get("duration_scale", 1.0)
chunk_silence = args.chunk_silence or config.get("chunk_silence", 0.0)
seed = args.seed if args.seed is not None else config.get("seed", None)
device = args.device or config.get("device", device)

if not srp_ckpt_file:
    raise ValueError(
        "CLF5-2 inference requires an SRP checkpoint. Pass --srp_ckpt_file or set srp_ckpt_file in the config."
    )


# patches for pip pkg user: resolve packaged example paths, but never clobber a
# path that already points to an existing file (avoids double-join issues).
def _resolve_example(path):
    if not path or "infer/examples/" not in path or os.path.exists(path):
        return path
    resolved = str(files("clf5_2").joinpath(path))
    return resolved if os.path.exists(resolved) else path


ref_audio = _resolve_example(ref_audio)
gen_file = _resolve_example(gen_file)
if "voices" in config:
    for voice in config["voices"]:
        config["voices"][voice]["ref_audio"] = _resolve_example(config["voices"][voice]["ref_audio"])


# ignore gen_text if gen_file provided

if gen_file:
    gen_text = codecs.open(gen_file, "r", "utf-8").read()


# output path

wave_path = Path(output_dir) / output_file
if save_chunk:
    output_chunk_dir = os.path.join(output_dir, f"{Path(output_file).stem}_chunks")
    if not os.path.exists(output_chunk_dir):
        os.makedirs(output_chunk_dir)


# load vocoder

if vocoder_name == "vocos":
    vocoder_local_path = "ckpts/vocos-mel-24khz"
elif vocoder_name == "bigvgan":
    vocoder_local_path = "ckpts/bigvgan_v2_24khz_100band_256x"

vocoder = load_vocoder(
    vocoder_name=vocoder_name, is_local=load_vocoder_from_local, local_path=vocoder_local_path, device=device
)


# load model config

model_cfg = OmegaConf.load(
    args.model_cfg or config.get("model_cfg", str(files("clf5_2").joinpath(f"configs/{model}.yaml")))
)
model_cls = get_class(f"clf5_2.model.{model_cfg.model.backbone}")
model_arc = model_cfg.model.arch

tokenizer = model_cfg.model.tokenizer
dataset_name = model_cfg.datasets.name

mel_cfg = model_cfg.model.mel_spec
target_sample_rate = mel_cfg.target_sample_rate
n_mel_channels = mel_cfg.n_mel_channels
hop_length = mel_cfg.hop_length
win_length = mel_cfg.win_length
n_fft = mel_cfg.n_fft

if vocoder_name != mel_cfg.mel_spec_type:
    raise ValueError(f"Vocoder {vocoder_name} does not match model mel spec type {mel_cfg.mel_spec_type}.")


# load speaking rate predictor (SRP)

model_sp = load_srp_model(srp_model_cfg_file, srp_ckpt_file, device)
model_sp.eval()


# load TTS model

if vocab_file:
    vocab_char_map, vocab_size = get_tokenizer(vocab_file, "custom")
else:
    vocab_char_map, vocab_size = get_tokenizer(dataset_name, tokenizer)

# prompt_token adds one extra learnable token id on top of the vocab
prompt_token = bool(model_cfg.model.get("prompt_token", False))
if prompt_token:
    vocab_size += 1

model = CFM_SFT(
    transformer=model_cls(**model_arc, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
    mel_spec_kwargs=dict(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mel_channels=n_mel_channels,
        target_sample_rate=target_sample_rate,
        mel_spec_type=vocoder_name,
    ),
    odeint_kwargs=dict(method="euler"),
    vocab_char_map=vocab_char_map,
    prefix_text_pad=bool(model_cfg.model.get("prefix_text_pad", False)),
    prompt_token=prompt_token,
    eos_marker=model_cfg.model.get("eos_marker", None),
).to(device)

if not ckpt_file:
    raise ValueError(
        "CLF5-2 inference requires a model checkpoint. Pass --ckpt_file or set ckpt_file in the config."
    )

dtype = torch.float32 if vocoder_name == "bigvgan" else None
print(f"Using {model_cfg.model.name} (CLF5-2 transcript-free)...")
print(f"Checkpoint: {ckpt_file}")
model = load_checkpoint(model, ckpt_file, device, dtype=dtype, use_ema=True)
model.eval()

# mel extractor shared by SRP duration prediction and the acoustic condition
mel_spectrogram = MelSpec(
    n_fft=n_fft,
    hop_length=hop_length,
    win_length=win_length,
    n_mel_channels=n_mel_channels,
    target_sample_rate=target_sample_rate,
    mel_spec_type=vocoder_name,
)


def load_ref_mel(ref_audio_path):
    """Load reference audio, resample, loudness-normalize, and return (mel[1,T,d], ref_rms, ref_mel_len)."""
    audio, sr = torchaudio.load(ref_audio_path)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    ref_rms = torch.sqrt(torch.mean(torch.square(audio)))
    if ref_rms < target_rms:
        audio = audio * target_rms / ref_rms
    if sr != target_sample_rate:
        audio = torchaudio.transforms.Resample(sr, target_sample_rate)(audio)

    mel = mel_spectrogram(audio).squeeze(0)  # [d, T]
    ref_mel_len = mel.shape[-1]
    mel = mel.unsqueeze(0).permute(0, 2, 1)  # [1, T, d]
    return mel.to(device), ref_rms, ref_mel_len


def chunk_text_by_units(text, max_units):
    """Split text into chunks of at most `max_units` syllable-level units.

    Mirrors X-Voice's chunk_text_by_units: split on punctuation, then greedily
    pack whole sentences into a chunk until adding the next would exceed
    max_units. Unit counting reuses CLF5-2's `count()` (handles EN + ZH).
    """
    chunks = []
    current_chunk = ""
    sentences = re.split(r"(?<=[;:,.!?])\s+|(?<=[；：，。！？])", text)

    for sentence in sentences:
        if not sentence:
            continue
        if count(current_chunk) + count(sentence) <= max_units:
            current_chunk += sentence + " " if len(sentence[-1].encode("utf-8")) == 1 else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " " if len(sentence[-1].encode("utf-8")) == 1 else sentence

    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks or [text.strip()]


def _generate_chunk(ref_mel, ref_rms, ref_mel_len, pred_speed, gen_text_, local_speed):
    """Generate one chunk given a pre-computed SRP speed. Returns a numpy wave."""
    ref_mel_len_t = torch.tensor([ref_mel_len], dtype=torch.long, device=device)

    # estimate target duration from syllable-level unit count / speaking rate
    gt_num_unit = count(gen_text_)
    pred_duration = min(max(gt_num_unit / max(pred_speed, 1e-3) / local_speed, 2), 30)
    # optionally scale the predicted duration (e.g. 0.85 to slightly speed up)
    pred_duration = pred_duration * duration_scale
    gen_mel_len = int((pred_duration * target_sample_rate) / hop_length)
    total_mel_len = ref_mel_len + gen_mel_len
    total_mel_len_t = torch.tensor([total_mel_len], dtype=torch.long, device=device)

    # tokenize target text; the model inserts ratio-based prompt tokens + EOS internally
    if tokenizer == "pinyin":
        text_list = convert_char_to_pinyin([gen_text_])
    else:
        text_list = [gen_text_]

    with torch.inference_mode():
        generated, _ = model.sample(
            cond=ref_mel,
            text=text_list,
            duration=total_mel_len_t,
            lens=ref_mel_len_t,
            steps=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            seed=seed,
        )

    gen = generated[0][ref_mel_len:total_mel_len, :].unsqueeze(0)
    gen_mel_spec = gen.permute(0, 2, 1).to(torch.float32)
    if vocoder_name == "vocos":
        generated_wave = vocoder.decode(gen_mel_spec).cpu()
    elif vocoder_name == "bigvgan":
        generated_wave = vocoder(gen_mel_spec).squeeze(0).cpu()

    if ref_rms < target_rms:
        generated_wave = generated_wave * ref_rms / target_rms

    return generated_wave.squeeze().numpy()


def infer_one(ref_mel, ref_rms, ref_mel_len, gen_text_, local_speed):
    """Transcript-free CLF5-2 generation for one voice segment.

    Long text is split into chunks whose size is bounded by the SRP-predicted
    speaking rate (no ref_text needed), each chunk is synthesized separately,
    and the chunks are cross-fade concatenated. This keeps each generation
    within a sane duration and avoids the fixed per-utterance length cap
    compressing long inputs.
    """
    # predict speaking rate once from the prompt audio (silence-robust SRP)
    ref_mel_len_t = torch.tensor([ref_mel_len], dtype=torch.long, device=device)
    pred_speed = model_sp.predict_speed(audio=ref_mel, lens=ref_mel_len_t).item()

    # budget the remaining time (22s total, minus the prompt) at the predicted
    # rate to decide how many units fit per chunk
    prompt_seconds = ref_mel_len * hop_length / target_sample_rate
    remaining_seconds = max(22 - prompt_seconds, 3)
    max_units = max(int(pred_speed * remaining_seconds * local_speed), 1)

    gen_text_chunks = chunk_text_by_units(gen_text_, max_units)
    if len(gen_text_chunks) > 1:
        print(f"  speaking_rate={pred_speed:.2f}/s, max_units={max_units}, split into {len(gen_text_chunks)} chunks")

    chunk_waves = [
        _generate_chunk(ref_mel, ref_rms, ref_mel_len, pred_speed, chunk_text_, local_speed)
        for chunk_text_ in gen_text_chunks
    ]

    # concatenate the chunk waves
    if len(chunk_waves) == 1:
        return chunk_waves[0]

    # if a silence gap is requested, insert it between chunks (plain concat,
    # cross-fading across a silence gap makes no sense)
    if chunk_silence and chunk_silence > 0:
        silence = np.zeros(int(chunk_silence * target_sample_rate), dtype=chunk_waves[0].dtype)
        pieces = []
        for i, w in enumerate(chunk_waves):
            if i > 0:
                pieces.append(silence)
            pieces.append(w)
        return np.concatenate(pieces)

    # otherwise cross-fade concatenate
    if cross_fade_duration <= 0:
        return np.concatenate(chunk_waves)

    final_wave = chunk_waves[0]
    for next_wave in chunk_waves[1:]:
        cross_fade_samples = int(cross_fade_duration * target_sample_rate)
        cross_fade_samples = min(cross_fade_samples, len(final_wave), len(next_wave))
        if cross_fade_samples <= 0:
            final_wave = np.concatenate([final_wave, next_wave])
            continue
        prev_overlap = final_wave[-cross_fade_samples:]
        next_overlap = next_wave[:cross_fade_samples]
        fade_out = np.linspace(1, 0, cross_fade_samples)
        fade_in = np.linspace(0, 1, cross_fade_samples)
        cross_faded_overlap = prev_overlap * fade_out + next_overlap * fade_in
        final_wave = np.concatenate(
            [final_wave[:-cross_fade_samples], cross_faded_overlap, next_wave[cross_fade_samples:]]
        )
    return final_wave


# inference process


def main():
    main_voice = {"ref_audio": ref_audio}
    if "voices" not in config:
        voices = {"main": main_voice}
    else:
        voices = config["voices"]
        voices["main"] = main_voice

    # preload reference mels per voice
    for voice in voices:
        print("Voice:", voice)
        print("ref_audio ", voices[voice]["ref_audio"])
        ref_mel, ref_rms, ref_mel_len = load_ref_mel(voices[voice]["ref_audio"])
        voices[voice]["ref_mel"] = ref_mel
        voices[voice]["ref_rms"] = ref_rms
        voices[voice]["ref_mel_len"] = ref_mel_len

    generated_audio_segments = []
    reg1 = r"(?=\[\w+\])"
    chunks = re.split(reg1, gen_text)
    reg2 = r"\[(\w+)\]"
    for text in chunks:
        if not text.strip():
            continue
        match = re.match(reg2, text)
        if match:
            voice = match[1]
        else:
            print("No voice tag found, using main.")
            voice = "main"
        if voice not in voices:
            print(f"Voice {voice} not found, using main.")
            voice = "main"
        text = re.sub(reg2, "", text)
        gen_text_ = text.strip()
        if not gen_text_:
            continue
        local_speed = voices[voice].get("speed", speed)
        print(f"Voice: {voice}")
        audio_segment = infer_one(
            voices[voice]["ref_mel"],
            voices[voice]["ref_rms"],
            voices[voice]["ref_mel_len"],
            gen_text_,
            local_speed,
        )
        generated_audio_segments.append(audio_segment)

        if save_chunk:
            save_text = gen_text_
            if len(save_text) > 200:
                save_text = save_text[:200] + " ... "
            if use_legacy_text:
                save_text = unidecode(save_text)
            save_text = re.sub(r"[\\/:\0]", "_", save_text)
            sf.write(
                os.path.join(output_chunk_dir, f"{len(generated_audio_segments) - 1}_{save_text}.wav"),
                audio_segment,
                target_sample_rate,
            )

    if generated_audio_segments:
        final_wave = np.concatenate(generated_audio_segments)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(wave_path, "wb") as f:
            sf.write(f.name, final_wave, target_sample_rate)
            # Remove silence
            if remove_silence:
                remove_silence_for_generated_wav(f.name)
            print(f.name)


if __name__ == "__main__":
    main()

# Cross-Lingual F5-TTS 2 (CLF5-2) transcript-free inference:
#   clf52-infer -c src/clf5_2/infer/examples/clf52/clf52.toml
#   CUDA_VISIBLE_DEVICES=0 python -m clf5_2.infer.infer_cli -c src/clf5_2/infer/examples/clf52/clf52.toml