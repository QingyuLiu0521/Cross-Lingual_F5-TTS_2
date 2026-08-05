import os
import json
from importlib.resources import files
from typing import Literal

import torch
import torch.nn.functional as F
import torchaudio
from datasets import Dataset as Dataset_
from datasets import load_from_disk
from torch import nn
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm
from pathlib import Path

from clf5_2.model.modules import MelSpec
from clf5_2.model.utils import default


class HFDataset(Dataset):
    def __init__(
        self,
        hf_dataset: Dataset,
        target_sample_rate=24_000,
        n_mel_channels=100,
        hop_length=256,
        n_fft=1024,
        win_length=1024,
        mel_spec_type="vocos",
    ):
        self.data = hf_dataset
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length

        self.mel_spectrogram = MelSpec(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        )

    def get_frame_len(self, index):
        row = self.data[index]
        audio = row["audio"]["array"]
        sample_rate = row["audio"]["sampling_rate"]
        return audio.shape[-1] / sample_rate * self.target_sample_rate / self.hop_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        audio = row["audio"]["array"]

        # logger.info(f"Audio shape: {audio.shape}")

        sample_rate = row["audio"]["sampling_rate"]
        duration = audio.shape[-1] / sample_rate

        if duration > 30 or duration < 0.3:
            return self.__getitem__((index + 1) % len(self.data))

        audio_tensor = torch.from_numpy(audio).float()

        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
            audio_tensor = resampler(audio_tensor)

        audio_tensor = audio_tensor.unsqueeze(0)  # 't -> 1 t')

        mel_spec = self.mel_spectrogram(audio_tensor)

        mel_spec = mel_spec.squeeze(0)  # '1 d t -> d t'

        text = row["text"]

        return dict(
            mel_spec=mel_spec,
            text=text,
        )


class CustomDataset(Dataset):
    def __init__(
        self,
        custom_dataset: Dataset,
        durations=None,
        target_sample_rate=24_000,
        hop_length=256,
        n_mel_channels=100,
        n_fft=1024,
        win_length=1024,
        mel_spec_type="vocos",
        preprocessed_mel=False,
        mel_spec_module: nn.Module | None = None,
    ):
        self.data = custom_dataset
        self.durations = durations
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.win_length = win_length
        self.mel_spec_type = mel_spec_type
        self.preprocessed_mel = preprocessed_mel

        if not preprocessed_mel:
            self.mel_spectrogram = default(
                mel_spec_module,
                MelSpec(
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    n_mel_channels=n_mel_channels,
                    target_sample_rate=target_sample_rate,
                    mel_spec_type=mel_spec_type,
                ),
            )

    def get_frame_len(self, index):
        if (
            self.durations is not None
        ):  # Please make sure the separately provided durations are correct, otherwise 99.99% OOM
            return self.durations[index] * self.target_sample_rate / self.hop_length
        return self.data[index]["duration"] * self.target_sample_rate / self.hop_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        while True:
            row = self.data[index]
            audio_path = row["audio_path"]
            text = row["text"]
            duration = row["duration"]

            # filter by given length
            if 0.3 <= duration <= 30:
                break  # valid

            index = (index + 1) % len(self.data)

        if self.preprocessed_mel:
            mel_spec = torch.tensor(row["mel_spec"])
        else:
            audio, source_sample_rate = torchaudio.load(audio_path)

            # make sure mono input
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)

            # resample if necessary
            if source_sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(source_sample_rate, self.target_sample_rate)
                audio = resampler(audio)

            # to mel spectrogram
            mel_spec = self.mel_spectrogram(audio)
            mel_spec = mel_spec.squeeze(0)  # '1 d t -> d t'

        return {
            "mel_spec": mel_spec,
            "text": text,
        }

class CustomDataset_gp(Dataset):
    def __init__(
        self,
        custom_dataset: Dataset,
        durations=None,
        target_sample_rate=24_000,
        hop_length=256,
        n_mel_channels=100,
        n_fft=1024,
        win_length=1024,
        mel_spec_type="vocos",
        preprocessed_mel=False,
        mel_spec_module: nn.Module | None = None,
    ):
        self.data = custom_dataset
        self.durations = durations
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.win_length = win_length
        self.mel_spec_type = mel_spec_type
        self.preprocessed_mel = preprocessed_mel

        if not preprocessed_mel:
            self.mel_spectrogram = default(
                mel_spec_module,
                MelSpec(
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    n_mel_channels=n_mel_channels,
                    target_sample_rate=target_sample_rate,
                    mel_spec_type=mel_spec_type,
                ),
            )

    def get_frame_len(self, index):
        if (
            self.durations is not None
        ):  # Please make sure the separately provided durations are correct, otherwise 99.99% OOM
            return self.durations[index] * self.target_sample_rate / self.hop_length
        return self.data[index]["duration"] * self.target_sample_rate / self.hop_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        while True:
            row = self.data[index]
            audio_path = row["audio_path"]
            text = row["text"]
            duration = row["duration"]
            rel_path = row["rel_path"]

            # filter by given length
            if 0.1 <= duration <= 50:
                break  # valid

            index = (index + 1) % len(self.data)

        if self.preprocessed_mel:
            mel_spec = torch.tensor(row["mel_spec"])
        else:
            audio, source_sample_rate = torchaudio.load(audio_path)

            # make sure mono input
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)

            # resample if necessary
            if source_sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(source_sample_rate, self.target_sample_rate)
                audio = resampler(audio)

            # to mel spectrogram
            mel_spec = self.mel_spectrogram(audio)
            mel_spec = mel_spec.squeeze(0)  # '1 d t -> d t'

        return {
            "mel_spec": mel_spec,
            "text": text,
            "rel_path": rel_path
        }

class CustomDataset_sft(Dataset):
    def __init__(
        self,
        custom_dataset: Dataset,
        durations=None,
        prompt_frames=None,
        target_sample_rate=24_000,
        hop_length=256,
        n_mel_channels=100,
        n_fft=1024,
        win_length=1024,
        mel_spec_type="vocos",
        preprocessed_mel=False,
        mel_spec_module: nn.Module | None = None,
    ):
        self.data = custom_dataset
        self.durations = durations
        self.prompt_frames = prompt_frames
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.win_length = win_length
        self.mel_spec_type = mel_spec_type
        self.preprocessed_mel = preprocessed_mel

        if not preprocessed_mel:
            self.mel_spectrogram = default(
                mel_spec_module,
                MelSpec(
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    n_mel_channels=n_mel_channels,
                    target_sample_rate=target_sample_rate,
                    mel_spec_type=mel_spec_type,
                ),
            )

    # def get_frame_len(self, index):
    #     prompt_frames = self.data[index]["prompt_frames"]
    #     if (
    #         self.durations is not None
    #     ):  # Please make sure the separately provided durations are correct, otherwise 99.99% OOM
    #         orig_frames = self.durations[index] * self.target_sample_rate / self.hop_length
    #         return orig_frames + prompt_frames
    #     orig_frames = self.data[index]["duration"] * self.target_sample_rate / self.hop_length
    #     return orig_frames + prompt_frames

    def get_frame_len(self, index):
        if self.durations is not None and self.prompt_frames is not None:
            orig_frames = self.durations[index] * self.target_sample_rate / self.hop_length
            return orig_frames + self.prompt_frames[index]
        elif (
            self.durations is not None
        ):  # Please make sure the separately provided durations are correct, otherwise 99.99% OOM
            prompt_frames = self.data[index]["prompt_frames"]
            orig_frames = self.durations[index] * self.target_sample_rate / self.hop_length
            return orig_frames + prompt_frames
        prompt_frames = self.data[index]["prompt_frames"]
        orig_frames = self.data[index]["duration"] * self.target_sample_rate / self.hop_length
        return orig_frames + prompt_frames

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        while True:
            row = self.data[index]
            audio_path = row["audio_path"]
            text = row["text"]
            total_text = row["total_text"]
            duration = row["duration"]
            mel_path = row["mel_path"]
            prompt_frames = row["prompt_frames"]

            target_frames = duration * self.target_sample_rate / self.hop_length
            total_frames = prompt_frames + target_frames

            # filter by given length
            if 0.3 <= duration <= 30 and total_frames <= 4096:
                break  # valid

            index = (index + 1) % len(self.data)

        prompt_mel = torch.load(mel_path, map_location='cpu', weights_only=True)
        prompt_mel = prompt_mel.transpose(0, 1) # (T, N) -> (N, T)
        if self.preprocessed_mel:
            mel_spec = torch.tensor(row["mel_spec"])
        else:
            audio, source_sample_rate = torchaudio.load(audio_path)

            # make sure mono input
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)

            # resample if necessary
            if source_sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(source_sample_rate, self.target_sample_rate)
                audio = resampler(audio)

            # to mel spectrogram
            mel_spec = self.mel_spectrogram(audio)
            mel_spec = mel_spec.squeeze(0)  # '1 d t -> d t'

        if prompt_mel.shape[0] != mel_spec.shape[0]:
            raise ValueError(f"Shape mismatch: prompt_mel shape[0] is {prompt_mel.shape[0]}, but target_mel shape[0] is {mel_spec.shape[0]}")

        return {
            "mel_spec": mel_spec,
            "text": text,
            "total_text": total_text,
            "prompt_mel": prompt_mel
        }

# Dynamic Batch Sampler
class DynamicBatchSampler(Sampler[list[int]]):
    """Extension of Sampler that will do the following:
    1.  Change the batch size (essentially number of sequences)
        in a batch to ensure that the total number of frames are less
        than a certain threshold.
    2.  Make sure the padding efficiency in the batch is high.
    3.  Shuffle batches each epoch while maintaining reproducibility.
    """

    def __init__(
        self, sampler: Sampler[int], frames_threshold: int, max_samples=0, random_seed=None, drop_residual: bool = False, drop_last: bool = True
    ):
        self.sampler = sampler
        self.frames_threshold = frames_threshold
        self.max_samples = max_samples
        self.random_seed = random_seed
        self.epoch = 0

        indices, batches = [], []
        data_source = self.sampler.data_source

        for idx in tqdm(
            self.sampler, desc="Sorting with sampler... if slow, check whether dataset is provided with duration"
        ):
            indices.append((idx, data_source.get_frame_len(idx)))
        indices.sort(key=lambda elem: elem[1])

        batch = []
        batch_frames = 0
        for idx, frame_len in tqdm(
            indices, desc=f"Creating dynamic batches with {frames_threshold} audio frames per gpu"
        ):
            if batch_frames + frame_len <= self.frames_threshold and (max_samples == 0 or len(batch) < max_samples):
                batch.append(idx)
                batch_frames += frame_len
            else:
                if len(batch) > 0:
                    batches.append(batch)
                if frame_len <= self.frames_threshold:
                    batch = [idx]
                    batch_frames = frame_len
                else:
                    batch = []
                    batch_frames = 0

        if not drop_residual and len(batch) > 0:
            batches.append(batch)

        del indices
        self.batches = batches

        # Ensure even batches with accelerate BatchSamplerShard cls under frame_per_batch setting
        self.drop_last = drop_last

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for this sampler."""
        self.epoch = epoch

    def __iter__(self):
        # Use both random_seed and epoch for deterministic but different shuffling per epoch
        if self.random_seed is not None:
            g = torch.Generator()
            g.manual_seed(self.random_seed + self.epoch)
            # Use PyTorch's random permutation for better reproducibility across PyTorch versions
            indices = torch.randperm(len(self.batches), generator=g).tolist()
            batches = [self.batches[i] for i in indices]
        else:
            batches = self.batches
        return iter(batches)

    def __len__(self):
        return len(self.batches)


# Load dataset


def load_dataset(
    dataset_name: str,
    tokenizer: str = "pinyin",
    dataset_type: str = "CustomDataset",
    audio_type: str = "raw",
    mel_spec_module: nn.Module | None = None,
    mel_spec_kwargs: dict = dict(),
    sft: bool = False,
    target_hour: int | None = None,
) -> CustomDataset | HFDataset:
    """
    dataset_type    - "CustomDataset" if you want to use tokenizer name and default data path to load for train_dataset
                    - "CustomDatasetPath" if you just want to pass the full path to a preprocessed dataset without relying on tokenizer
    sft             - when True, load the synthetic-prompt SFT dataset (data dir suffix "_sft", carries prompt_frames,
                      builds CustomDataset_sft); when False, load the base CustomDataset
    """

    print("Loading dataset ...")

    if dataset_type == "CustomDataset":
        if not sft:
            rel_data_path = str(files("clf5_2").joinpath(f"../../data/{dataset_name}_{tokenizer}"))
            if audio_type == "raw":
                try:
                    train_dataset = load_from_disk(f"{rel_data_path}/raw")
                except:  # noqa: E722
                    train_dataset = Dataset_.from_file(f"{rel_data_path}/raw.arrow")
                preprocessed_mel = False
            elif audio_type == "mel":
                train_dataset = Dataset_.from_file(f"{rel_data_path}/mel.arrow")
                preprocessed_mel = True
            with open(f"{rel_data_path}/duration.json", "r", encoding="utf-8") as f:
                data_dict = json.load(f)
            durations = data_dict["duration"]
            train_dataset = CustomDataset(
                train_dataset,
                durations=durations,
                preprocessed_mel=preprocessed_mel,
                mel_spec_module=mel_spec_module,
                **mel_spec_kwargs,
            )
        else:
            if target_hour is None:
                rel_data_path = str(files("clf5_2").joinpath(f"../../data/{dataset_name}_{tokenizer}_sft"))
            else:
                rel_data_path = str(files("clf5_2").joinpath(f"../../data/{dataset_name}_{tokenizer}_sft_hours{target_hour}"))
            if audio_type == "raw":
                try:
                    train_dataset = load_from_disk(f"{rel_data_path}/raw")
                except:  # noqa: E722
                    train_dataset = Dataset_.from_file(f"{rel_data_path}/raw.arrow")
                preprocessed_mel = False
            elif audio_type == "mel":
                train_dataset = Dataset_.from_file(f"{rel_data_path}/mel.arrow")
                preprocessed_mel = True
            with open(f"{rel_data_path}/duration.json", "r", encoding="utf-8") as f:
                data_dict = json.load(f)
            durations = data_dict["duration"]
            prompt_frames = data_dict.get("prompt_frames", None)
            train_dataset = CustomDataset_sft(
                train_dataset,
                durations=durations,
                prompt_frames=prompt_frames,
                preprocessed_mel=preprocessed_mel,
                mel_spec_module=mel_spec_module,
                **mel_spec_kwargs,
            )

    elif dataset_type == "CustomDatasetPath":
        try:
            train_dataset = load_from_disk(f"{dataset_name}/raw")
        except:  # noqa: E722
            train_dataset = Dataset_.from_file(f"{dataset_name}/raw.arrow")

        with open(f"{dataset_name}/duration.json", "r", encoding="utf-8") as f:
            data_dict = json.load(f)
        durations = data_dict["duration"]
        train_dataset = CustomDataset(
            train_dataset, durations=durations, preprocessed_mel=preprocessed_mel, **mel_spec_kwargs
        )

    elif dataset_type == "HFDataset":
        print(
            "Should manually modify the path of huggingface dataset to your need.\n"
            + "May also the corresponding script cuz different dataset may have different format."
        )
        pre, post = dataset_name.split("_")
        train_dataset = HFDataset(
            load_dataset(f"{pre}/{pre}", split=f"train.{post}", cache_dir=str(files("clf5_2").joinpath("../../data"))),
        )

    return train_dataset

def load_dataset_gp(
    dataset_name: str,
    tokenizer: str = "pinyin",
    dataset_type: str = "CustomDataset",
    audio_type: str = "raw",
    mel_spec_module: nn.Module | None = None,
    mel_spec_kwargs: dict = dict(),
) -> CustomDataset | HFDataset:
    """
    dataset_type    - "CustomDataset" if you want to use tokenizer name and default data path to load for train_dataset
                    - "CustomDatasetPath" if you just want to pass the full path to a preprocessed dataset without relying on tokenizer
    """

    print("Loading dataset ...")

    if dataset_type == "CustomDataset":
        rel_data_path = str(files("clf5_2").joinpath(f"../../data/{dataset_name}_{tokenizer}_gp_inference"))
        if audio_type == "raw":
            try:
                train_dataset = load_from_disk(f"{rel_data_path}/raw")
            except:  # noqa: E722
                train_dataset = Dataset_.from_file(f"{rel_data_path}/raw.arrow")
            preprocessed_mel = False
        elif audio_type == "mel":
            train_dataset = Dataset_.from_file(f"{rel_data_path}/mel.arrow")
            preprocessed_mel = True
        with open(f"{rel_data_path}/duration.json", "r", encoding="utf-8") as f:
            data_dict = json.load(f)
        durations = data_dict["duration"]
        train_dataset = CustomDataset_gp(
            train_dataset,
            durations=durations,
            preprocessed_mel=preprocessed_mel,
            mel_spec_module=mel_spec_module,
            **mel_spec_kwargs,
        )

    elif dataset_type == "CustomDatasetPath":
        try:
            train_dataset = load_from_disk(f"{dataset_name}/raw")
        except:  # noqa: E722
            train_dataset = Dataset_.from_file(f"{dataset_name}/raw.arrow")

        with open(f"{dataset_name}/duration.json", "r", encoding="utf-8") as f:
            data_dict = json.load(f)
        durations = data_dict["duration"]
        train_dataset = CustomDataset_gp(
            train_dataset, durations=durations, preprocessed_mel=preprocessed_mel, **mel_spec_kwargs
        )

    elif dataset_type == "HFDataset":
        print(
            "Should manually modify the path of huggingface dataset to your need.\n"
            + "May also the corresponding script cuz different dataset may have different format."
        )
        pre, post = dataset_name.split("_")
        train_dataset = HFDataset(
            load_dataset(f"{pre}/{pre}", split=f"train.{post}", cache_dir=str(files("clf5_2").joinpath("../../data"))),
        )

    return train_dataset


# collation


def collate_fn(batch):
    mel_specs = [item["mel_spec"].squeeze(0) for item in batch]
    mel_lengths = torch.LongTensor([spec.shape[-1] for spec in mel_specs])
    max_mel_length = mel_lengths.amax()

    padded_mel_specs = []
    for spec in mel_specs:
        padding = (0, max_mel_length - spec.size(-1))
        padded_spec = F.pad(spec, padding, value=0)
        padded_mel_specs.append(padded_spec)

    mel_specs = torch.stack(padded_mel_specs)

    text = [item["text"] for item in batch]
    text_lengths = torch.LongTensor([len(item) for item in text])

    return dict(
        mel=mel_specs,
        mel_lengths=mel_lengths,  # records for padding mask
        text=text,
        text_lengths=text_lengths,
    )

def collate_fn_gp_inference(batch):
    mel_specs = [item["mel_spec"].squeeze(0) for item in batch]
    mel_lengths = torch.LongTensor([spec.shape[-1] for spec in mel_specs])
    max_mel_length = mel_lengths.amax()

    padded_mel_specs = []
    for spec in mel_specs:  # TODO. maybe records mask for attention here
        padding = (0, max_mel_length - spec.size(-1))
        padded_spec = F.pad(spec, padding, value=0)
        padded_mel_specs.append(padded_spec)

    mel_specs = torch.stack(padded_mel_specs)

    text = [item["text"] for item in batch]
    text_lengths = torch.LongTensor([len(item) for item in text])
    rel_paths = [item["rel_path"] for item in batch]

    return dict(
        mel=mel_specs,
        mel_lengths=mel_lengths,
        text=text,
        text_lengths=text_lengths,
        rel_paths=rel_paths,
    )

def collate_fn_sft(batch):
    mel_specs = []
    prompt_mel_lengths = []

    for item in batch:
        prompt = item['prompt_mel']  # Shape: (N, T_prompt)
        target = item['mel_spec']    # Shape: (N, T_target)
        
        combined_mel = torch.cat([prompt, target], dim=1) # Shape: (N, T_total)
        mel_specs.append(combined_mel)
        
        prompt_mel_lengths.append(prompt.shape[1]) 

    mel_lengths = torch.LongTensor([spec.shape[1] for spec in mel_specs])
    max_mel_length = mel_lengths.amax()

    padded_mel_specs = []
    for spec in mel_specs:
        padding = (0, max_mel_length - spec.size(-1))
        padded_spec = F.pad(spec, padding, value=0)
        padded_mel_specs.append(padded_spec)

    mel_specs = torch.stack(padded_mel_specs)

    text = [item["text"] for item in batch]
    text_lengths = torch.LongTensor([len(item) for item in text])

    total_text = [item["total_text"] for item in batch]
    total_text_lengths = torch.LongTensor([len(item) for item in total_text])
    
    prompt_mel_lengths = torch.LongTensor(prompt_mel_lengths)

    return dict(
        mel=mel_specs,
        mel_lengths=mel_lengths,  # records for padding mask
        text=text,
        text_lengths=text_lengths,
        total_text=total_text,
        total_text_lengths=total_text_lengths,
        prompt_mel_lengths=prompt_mel_lengths,
    )

class CustomDataset_eval_metric(Dataset):
    def __init__(
        self,
        custom_dataset: Dataset,
        durations=None,
        prompt_frames=None,
        target_sample_rate=24_000,
        hop_length=256,
        metric="utmos", # newly added metric parameter
        eval_sample_rate=16_000, # newly added evaluation sample rate
    ):
        self.data = custom_dataset
        self.durations = durations
        self.prompt_frames = prompt_frames
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        self.metric = metric
        self.eval_sample_rate = eval_sample_rate

    def get_frame_len(self, index):
        if self.metric == "utmos":
            # UTMOS only evaluates the generated audio, sort by the generated mel length
            if self.prompt_frames is not None:
                return self.prompt_frames[index]
            return self.data[index]["prompt_frames"]

        elif self.metric == "sim":
            # SIM needs to load the original audio, sort by the original audio length
            if self.durations is not None:
                return self.durations[index] * self.target_sample_rate / self.hop_length
            return self.data[index]["duration"] * self.target_sample_rate / self.hop_length
        # by default return the number of generated mel frames
        if self.prompt_frames is not None:
            return self.prompt_frames[index]
        return self.data[index]["prompt_frames"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        
        # 1. Load the generated Mel (CPU)
        mel_path = row["mel_path"]
        gen_mel = torch.load(mel_path, map_location='cpu', weights_only=True)
        # (T, Mel) -> (Mel, T)
        gen_mel = gen_mel.transpose(0, 1)

        result = {
            "gen_mel": gen_mel,
            "mel_path_stem": str(Path(mel_path).with_suffix("")), # strip .pt
        }

        # 2. If in SIM mode, load the reference audio
        if self.metric == "sim":
            audio_path = row["audio_path"]
            try:
                wav_ref, sr = torchaudio.load(audio_path)
                # convert to mono
                if wav_ref.shape[0] > 1:
                    wav_ref = torch.mean(wav_ref, dim=0, keepdim=True)
                # resample to 16k (for evaluation)
                if sr != self.eval_sample_rate:
                    resampler = torchaudio.transforms.Resample(sr, self.eval_sample_rate)
                    wav_ref = resampler(wav_ref)
                result["ref_audio"] = wav_ref.squeeze(0) # (T,)
            except Exception as e:
                print(f"Error loading {audio_path}: {e}")
                # return an all-zero tensor to prevent a crash, length set to 16000 (1s)
                result["ref_audio"] = torch.zeros(16000)

        return result

def collate_fn_eval_metric(batch):
    # 1. Pad Generated Mel
    gen_mels = [item["gen_mel"] for item in batch]
    mel_lengths = torch.LongTensor([spec.shape[-1] for spec in gen_mels])
    max_mel_len = mel_lengths.amax()
    
    padded_mels = []
    for spec in gen_mels:
        padding = (0, max_mel_len - spec.size(-1))
        padded_spec = F.pad(spec, padding, value=0)
        padded_mels.append(padded_spec)
    padded_mels = torch.stack(padded_mels) # (B, Mel, T)

    mel_path_stems = [item["mel_path_stem"] for item in batch]

    batch_dict = {
        "gen_mel": padded_mels,
        "mel_lengths": mel_lengths,
        "mel_path_stems": mel_path_stems
    }

    # 2. Pad Reference Audio (Sim mode only)
    if "ref_audio" in batch[0]:
        ref_audios = [item["ref_audio"] for item in batch]
        ref_lengths = torch.LongTensor([wav.shape[-1] for wav in ref_audios])
        max_audio_len = ref_lengths.amax()
        
        padded_refs = []
        for wav in ref_audios:
            padding = (0, max_audio_len - wav.size(-1))
            padded_wav = F.pad(wav, padding, value=0)
            padded_refs.append(padded_wav)
            
        padded_refs = torch.stack(padded_refs) # (B, T)
        batch_dict["ref_audio"] = padded_refs
        batch_dict["ref_lengths"] = ref_lengths

    return batch_dict

def load_dataset_eval_metric(
    dataset_name: str,
    tokenizer: str = "pinyin",
    dataset_type: str = "CustomDataset",
    target_sample_rate: int = 24000,
    hop_length: int = 256,
    metric: str = "utmos" # metric passed in
) -> CustomDataset_eval_metric:
    
    print(f"Loading dataset {dataset_name} for {metric} evaluation ...")
    
    if dataset_type == "CustomDataset":
        rel_data_path = str(files("clf5_2").joinpath(f"../../data/{dataset_name}_{tokenizer}_gp_eval"))
        try:
            raw_dataset = load_from_disk(f"{rel_data_path}/raw")
        except:
            raw_dataset = Dataset_.from_file(f"{rel_data_path}/raw.arrow")
        
        with open(f"{rel_data_path}/duration.json", "r", encoding="utf-8") as f:
            data_dict = json.load(f)
        durations = data_dict["duration"]
        prompt_frames = data_dict.get("prompt_frames", None)
        
        eval_dataset = CustomDataset_eval_metric(
            raw_dataset,
            durations=durations,
            prompt_frames=prompt_frames,
            target_sample_rate=target_sample_rate,
            hop_length=hop_length,
            metric=metric # pass the parameter
        )
        return eval_dataset
    else:
        raise ValueError(f"Unsupported dataset_type: {dataset_type}")