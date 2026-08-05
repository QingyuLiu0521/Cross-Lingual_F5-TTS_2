# Cross-Lingual F5-TTS 2: A Simplified Framework for Language-Agnostic Voice Cloning

<a href="https://huggingface.co/QingyuLiu1/Cross-Lingual_F5-TTS_2" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/Models-yellow?labelColor=grey&logo=huggingface&style=for-the-badge" alt="HF Model"></a>
<a href="https://qingyuliu0521.github.io/Cross-Lingual_F5-TTS_2_demo/" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/-Demo%20Page-orange?labelColor=grey&logo=github&style=for-the-badge" alt="Demo"></a>
<img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">

**Cross-Lingual F5-TTS 2 (CLF5-2)** is a forced-alignment-free fine-tuning framework that turns a pretrained F5-TTS into a **transcript-free, cross-lingual, zero-shot** text-to-speech system. It clones a speaker from a short reference clip in *any* language and synthesizes English or Chinese speech, **without ever needing the transcript of the audio prompt**.

## News

- **2026/08**: Cross-Lingual F5-TTS 2 code and [models](https://huggingface.co/QingyuLiu1/Cross-Lingual_F5-TTS_2) are released.

## Installation

### Create a separate environment

```bash
# Create a conda env with python>=3.10
conda create -n clf52 python=3.10
conda activate clf52
```

### Install PyTorch matched to your device

Install a PyTorch build that matches your machine (CUDA / CPU) **before** installing this package, e.g.:

```bash
# example: CUDA 12.1 build
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Install Cross-Lingual F5-TTS 2

```bash
pip install -e .

# optional extras for evaluation
# (faster_whisper, funasr, jiwer, modelscope, zhconv, zhon)
pip install -e ".[eval]"
```

### Download checkpoints

Download the checkpoints from the Hugging Face repo [QingyuLiu1/Cross-Lingual_F5-TTS_2](https://huggingface.co/QingyuLiu1/Cross-Lingual_F5-TTS_2) and place them under `ckpts/`:

```text
ckpts/
├── clf52_100000.safetensors        # main CLF5-2 (synthetic-prompt SFT) model
└── silence-robust-srp.safetensors  # silence-robust speaking rate predictor
```

## Inference

Run zero-shot voice cloning with the packaged example config:

```bash
# console script (installed by `pip install -e .`)
clf52-infer -c src/clf5_2/infer/examples/clf52/clf52.toml

# equivalent module form
python -m clf5_2.infer.infer_cli -c src/clf5_2/infer/examples/clf52/clf52.toml
```

## Training

CLF5-2 is trained in two stages. See the [training guidance](src/clf5_2/train/README.md) for data preparation, configs, and W&B logging.

### Speaking Rate Predictor Training

Refer to the [speaking rate predictor guidance](src/rate_pred/README.md) for the silence-robust speaking rate predictor
 used for transcript-free duration prediction.

## Evaluation

Refer to the [evaluation guidance](src/clf5_2/eval/README.md) for benchmark and metric (WER / SIM / UTMOS) scripts.

## Repo Structure

```text
Cross-Lingual_F5-TTS_2/
├── ckpts/                  # checkpoints (see ckpts/README.md)
├── data/                   # datasets and processed data
├── src/
│   ├── clf5_2/             # main transcript-free TTS package
│   └── rate_pred/          # speaking rate predictor (SRP)
└── pyproject.toml          # package definition and dependencies
```

## Acknowledgements

- [F5-TTS](https://arxiv.org/abs/2410.06885) by SWivid — the brilliant base model and foundation of this codebase.
- **Cross-Lingual F5-TTS** (Proc. ICASSP 2026) — the predecessor whose syllable-level speaking rate predictor CLF5-2 builds on.
- [Vocos](https://huggingface.co/charactr/vocos-mel-24khz) as the mel vocoder (BigVGAN is also supported as an alternative vocoder).
- [torchdiffeq](https://github.com/rtqichen/torchdiffeq) as the ODE solver.
- [FunASR](https://github.com/modelscope/FunASR) and [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for evaluation tools.

## Citations

If you find our work useful, please cite:

```bibtex
@article{liu2026crosslingualf5tts2,
  title   = {Cross-Lingual {F5-TTS} 2: A Simplified Framework for Language-Agnostic Voice Cloning},
  author  = {Liu, Qingyu and Xu, Rixi and Chen, Yushen and Niu, Zhikang and Li, Haitao and Zhu, Pengcheng and Zhang, Bowen and Zhao, Jian and Yang, Yunting and Cheng, Qinyuan and Qiu, Xipeng and Sisman, Berrak and Yu, Kai and Chen, Xie},
  year    = {2026}
}
```

## License

Code is released under the MIT License.
