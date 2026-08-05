# ruff: noqa: F722 F821

from __future__ import annotations

import os
import random
from collections import defaultdict
from importlib.resources import files

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import json
import jieba
from pypinyin import lazy_pinyin, Style
import math
import re
from typing import List

available_texts = [
    "The quick brown fox jumps.",
    "Life is what happens to you.",
    "Knowledge is power and wisdom is strength.",
    "Time flies when you are having fun.",
    "Practice makes perfect in everything.",
    "Where there is a will there is way.",
    "Actions speak louder than words do.",
    "The early bird catches the worm.",
    "Never give up on your dreams.",
    "Learning is a lifelong journey for everyone.",
    "Success comes to those who work hard.",
    "Every cloud has a silver lining.",
    "Rome was not built in a day.",
    "A journey begins with a single step.",
    "Believe in yourself and anything is possible.",
    "The best time to plant a tree.",
    "Happiness is a choice we make daily.",
    "Great minds think alike sometimes.",
    "Tomorrow is another day to try.",
    "Stay hungry and stay foolish always.",
]

# seed everything


def seed_everything(seed=0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# helpers


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


def is_package_available(package_name: str) -> bool:
    try:
        import importlib

        package_exists = importlib.util.find_spec(package_name) is not None
        return package_exists
    except Exception:
        return False


# tensor helpers


def lens_to_mask(t: int["b"], length: int | None = None) -> bool["b n"]:
    if not exists(length):
        length = t.amax()

    seq = torch.arange(length, device=t.device)
    return seq[None, :] < t[:, None]


def mask_from_start_end_indices(seq_len: int["b"], start: int["b"], end: int["b"]):
    max_seq_len = seq_len.max().item()
    seq = torch.arange(max_seq_len, device=start.device).long()
    start_mask = seq[None, :] >= start[:, None]
    end_mask = seq[None, :] < end[:, None]
    return start_mask & end_mask


def mask_from_frac_lengths(seq_len: int["b"], frac_lengths: float["b"]):
    lengths = (frac_lengths * seq_len).long()
    max_start = seq_len - lengths

    rand = torch.rand_like(frac_lengths)
    start = (max_start * rand).long().clamp(min=0)
    end = start + lengths

    return mask_from_start_end_indices(seq_len, start, end)

def mask_from_prompt_lens(prompt_lens: int["b"], lens: int["b"]):
    return mask_from_start_end_indices(lens, prompt_lens, lens)


def maybe_masked_mean(t: float["b n d"], mask: bool["b n"] = None) -> float["b d"]:
    if not exists(mask):
        return t.mean(dim=1)

    t = torch.where(mask[:, :, None], t, torch.tensor(0.0, device=t.device))
    num = t.sum(dim=1)
    den = mask.float().sum(dim=1)

    return num / den.clamp(min=1.0)


# simple utf-8 tokenizer, since paper went character based
def list_str_to_tensor(text: list[str], padding_value=-1) -> int["b nt"]:
    list_tensors = [torch.tensor([*bytes(t, "UTF-8")]) for t in text]  # ByT5 style
    text = pad_sequence(list_tensors, padding_value=padding_value, batch_first=True)
    return text


# char tokenizer, based on custom dataset's extracted .txt file
def list_str_to_idx(
    text: list[str] | list[list[str]]| list[list[int]],
    vocab_char_map: dict[str, int]| None,  # {char: idx}
    padding_value=-1,
) -> int["b nt"]:
    list_idx_tensors = [torch.tensor([vocab_char_map.get(c, 0) for c in t]) for t in text]  # pinyin or char style
    text = pad_sequence(list_idx_tensors, padding_value=padding_value, batch_first=True)
    return text

def bpe_padded(
    token_ids: list[list[int]],
    padding_value: int = -1,
) -> int["b nt"]:
    list_idx_tensors = [torch.tensor(ids) for ids in token_ids]
    padded = pad_sequence(list_idx_tensors, padding_value=padding_value, batch_first=True)
    return padded

# Get tokenizer


def get_tokenizer(dataset_name, tokenizer: str = "pinyin", sft: bool = False):
    """
    tokenizer   - "pinyin" do g2p for only chinese characters, need .txt vocab_file
                - "char" for char-wise tokenizer, need .txt vocab_file
                - "byte" for utf-8 tokenizer
                - "custom" if you're directly passing in a path to the vocab.txt you want to use
    vocab_size  - if use "pinyin", all available pinyin types, common alphabets (also those with accent) and symbols
                - if use "char", derived from unfiltered character & symbol counts of custom dataset
                - if use "byte", set to 256 (unicode byte range)
    sft         - when True, load from the synthetic-prompt SFT data dir (suffix "_sft"); when False, the base data dir
    """
    suffix = "_sft" if sft else ""
    if tokenizer in ["pinyin", "char"]:
        tokenizer_path = os.path.join(files("clf5_2").joinpath("../../data"), f"{dataset_name}_{tokenizer}{suffix}/vocab.txt")
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            vocab_char_map = {}
            for i, char in enumerate(f):
                vocab_char_map[char[:-1]] = i
        vocab_size = len(vocab_char_map)
        assert vocab_char_map[" "] == 0, "make sure space is of idx 0 in vocab.txt, cuz 0 is used for unknown char"

    elif tokenizer == "byte":
        vocab_char_map = None
        vocab_size = 256

    elif tokenizer == "bpe":
        vocab_char_map = None
        tokenizer_info_path = os.path.join(files("clf5_2").joinpath("../../data"), f"{dataset_name}_{tokenizer}{suffix}/tokenizer_info.json")
        with open(tokenizer_info_path, "r", encoding="utf-8") as f:
            tokenizer_info = json.load(f)
        vocab_size = tokenizer_info["vocab_size"]

    elif tokenizer == "custom":
        with open(dataset_name, "r", encoding="utf-8") as f:
            vocab_char_map = {}
            for i, char in enumerate(f):
                vocab_char_map[char[:-1]] = i
        vocab_size = len(vocab_char_map)

    return vocab_char_map, vocab_size


# convert char to pinyin


def convert_char_to_pinyin(text_list, polyphone=True):
    if jieba.dt.initialized is False:
        jieba.default_logger.setLevel(50)  # CRITICAL
        jieba.initialize()

    final_text_list = []
    custom_trans = str.maketrans(
        {";": ",", "“": '"', "”": '"', "‘": "'", "’": "'"}
    )  # add custom trans here, to address oov

    def is_chinese(c):
        return (
            "\u3100" <= c <= "\u9fff"  # common chinese characters
        )

    for text in text_list:
        char_list = []
        text = text.translate(custom_trans)
        for seg in jieba.cut(text):
            seg_byte_len = len(bytes(seg, "UTF-8"))
            if seg_byte_len == len(seg):  # if pure alphabets and symbols
                if char_list and seg_byte_len > 1 and char_list[-1] not in " :'\"":
                    char_list.append(" ")
                char_list.extend(seg)
            elif polyphone and seg_byte_len == 3 * len(seg):  # if pure east asian characters
                seg_ = lazy_pinyin(seg, style=Style.TONE3, tone_sandhi=True)
                for i, c in enumerate(seg):
                    if is_chinese(c):
                        char_list.append(" ")
                    char_list.append(seg_[i])
            else:  # if mixed characters, alphabets and symbols
                for c in seg:
                    if ord(c) < 256:
                        char_list.extend(c)
                    elif is_chinese(c):
                        char_list.append(" ")
                        char_list.extend(lazy_pinyin(c, style=Style.TONE3, tone_sandhi=True))
                    else:
                        char_list.append(c)
        final_text_list.append(char_list)

    return final_text_list


# filter func for dirty data with many repetitions


def repetition_found(text, length=2, tolerance=10):
    pattern_count = defaultdict(int)
    for i in range(len(text) - length + 1):
        pattern = text[i : i + length]
        pattern_count[pattern] += 1
    for pattern, count in pattern_count.items():
        if count > tolerance:
            return True
    return False


# get the empirically pruned step for sampling


def get_epss_timesteps(n, device, dtype):
    dt = 1 / 32
    predefined_timesteps = {
        5: [0, 2, 4, 8, 16, 32],
        6: [0, 2, 4, 6, 8, 16, 32],
        7: [0, 2, 4, 6, 8, 16, 24, 32],
        10: [0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32],
        12: [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32],
        16: [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32],
    }
    t = predefined_timesteps.get(n, [])
    if not t:
        return torch.linspace(0, 1, n + 1, device=device, dtype=dtype)
    return dt * torch.tensor(t, device=device, dtype=dtype)


def prefix_text_padding(
    text: int["b nt"],
    lens: int["b"] | None,
    prompt_lens: int["b"] | None,
    prefix_token_id: int = -1,
    anchor_token_ids: int["na"] | None = None,
) -> int["b nt"]:
    if not exists(prompt_lens):
        raise ValueError("prompt_lens is required when prefix_text_pad is enabled")

    if not exists(lens):
        raise ValueError("lens is required when prefix_text_pad is enabled")

    lens = lens.to(device=text.device, dtype=torch.long)
    prompt_lens = prompt_lens.to(device=text.device, dtype=torch.long)
    if exists(anchor_token_ids):
        anchor_token_ids = anchor_token_ids.to(device=text.device, dtype=text.dtype)
        if anchor_token_ids.ndim != 1:
            raise ValueError("anchor_token_ids must be a 1D tensor")

    if text.shape[0] != lens.shape[0] or text.shape[0] != prompt_lens.shape[0]:
        raise ValueError("text, lens, and prompt_lens must have the same batch size")
    if torch.any(prompt_lens > lens):
        raise ValueError("prompt_lens must be less than or equal to lens")

    padded_text = []
    for i in range(text.shape[0]):
        sample_text = text[i]
        target_text = sample_text[sample_text != -1]
        prompt_len = int(prompt_lens[i].item())
        total_len = int(lens[i].item())
        target_mel_len = total_len - prompt_len

        if prompt_len <= 0 or target_text.numel() == 0:
            padded_text.append(target_text)
            continue

        if target_mel_len <= 0:
            raise ValueError("lens must be greater than prompt_lens when prefix_text_pad is enabled")

        num_prefix = round(prompt_len / target_mel_len * int(target_text.numel()))
        if num_prefix <= 0:
            padded_text.append(target_text)
            continue

        prefix = torch.full((num_prefix,), prefix_token_id, device=text.device, dtype=text.dtype)
        if exists(anchor_token_ids):
            padded_text.append(torch.cat((prefix, anchor_token_ids, target_text), dim=0))
        else:
            padded_text.append(torch.cat((prefix, target_text), dim=0))

    return pad_sequence(padded_text, batch_first=True, padding_value=-1)

def create_derangement(n):
    """Create a derangement, ensuring no element stays in its original position"""
    if n == 1:
        raise ValueError("Cannot create derangement for single element")

    import random

    # Use Sattolo's algorithm - a variant of the Fisher-Yates shuffle
    # Guarantees that the generated permutation is a derangement
    perm = list(range(n))
    for i in range(n - 1, 0, -1):
        # Pick a random index in the range [0, i), excluding i
        j = random.randint(0, i - 1)
        perm[i], perm[j] = perm[j], perm[i]
    
    return perm

def trim_text(texts: List[str], frac_lengths: torch.Tensor) -> List[str]:
    """
    Trim a list of texts while keeping word/character boundaries intact

    Args:
        texts: list of strings
        frac_lengths: trimming ratio for each text (0.1~0.4)

    Returns:
        list of trimmed texts
    """
    
    # Use the same tokenization pattern as prefix_by_word_boundary_Emilia
    word_chars_unicode = (
        "a-zA-Z"         # basic Latin letters
        "\u00C0-\u02FF\u1E00-\u1EFF"  # Latin extended letters
        "\u0370-\u03FF\u1F00-\u1FFF"  # Greek letters
        "\u0400-\u04FF"  # Cyrillic letters
        "\u0590-\u05FF"  # Hebrew
        "\u0600-\u06FF"  # Arabic
    )
    other_symbols = "\u2211\u221A\u221E\u2260\u2264\u2192\u20B1\u20B9\u20BD\u2103"
    
    tokenization_pattern = re.compile(
        rf"[{word_chars_unicode}]+'[{word_chars_unicode}]+"  # words with an apostrophe
        r"|\d+\.\d+"            # decimals
        r"|\d+-\d+"             # numeric ranges
        r"|[a-zA-Z]+-\d+"       # letter-number
        r"|\d+-[a-zA-Z]+"       # number-letter
        rf"|[{word_chars_unicode}0-9\$€£¢¥%{other_symbols}]+"  # extended characters
        r"|[\u3100-\u9fff]"     # Chinese characters
        r"|[\u0E00-\u0E7F]"     # Thai
        r"|[\u3040-\u309F\u30A0-\u30FF]"  # Japanese
        r"|[\uAC00-\uD7A3\u1100-\u11FF\u3130-\u318F]"  # Korean
        r"|\s+"                 # whitespace
        r"|\S"                  # any other single character
    )
    
    # countable-unit pattern
    countable_pattern = re.compile(
        rf'^[{word_chars_unicode}]+(?:\'[{word_chars_unicode}]+)?$'
        r'|^[\u3100-\u9fff]$'   # single Chinese character
        r'|^[\u0E00-\u0E7F]$'   # single Thai character
        r'|^[\u3040-\u309F\u30A0-\u30FF]$'   # Japanese
        r'|^[\uAC00-\uD7A3\u1100-\u11FF\u3130-\u318F]$'   # Korean
    )
    
    processed_results = []
    
    for i, text in enumerate(texts):
        text = text.strip()
        if not text:
            processed_results.append("")
            continue
        
        # get the keep ratio
        keep_ratio = frac_lengths[i].item()  # convert to keep ratio
        
        # tokenize
        tokens = tokenization_pattern.findall(text)
        
        # count the total number of countable units
        countable_units = []
        countable_indices = []
        for idx, token in enumerate(tokens):
            if countable_pattern.fullmatch(token):
                countable_units.append(token)
                countable_indices.append(idx)
        
        if not countable_units:
            # no countable units, trim by characters
            keep_len = max(1, int(len(text) * keep_ratio))
            processed_results.append(text[:keep_len])
            continue
        
        # compute the number of countable units to keep
        keep_count = max(5, int(len(countable_units) * keep_ratio))
        
        # find the index of the last countable unit to keep
        if keep_count >= len(countable_units):
            # keep everything
            processed_results.append(text)
            continue
        
        # find the position in the original tokens of the last countable unit to keep
        last_keep_idx = countable_indices[keep_count - 1]
        
        # starting after that position, find the next non-whitespace character
        end_idx = last_keep_idx + 1
        while end_idx < len(tokens):
            token = tokens[end_idx]
            # if it is whitespace or punctuation, keep including it
            if not countable_pattern.fullmatch(token):
                # if it is a punctuation mark, include it
                if token.strip() and not token.isspace():
                    end_idx += 1
                    break
                end_idx += 1
            else:
                # reached the next countable unit, stop
                break
        
        # build the trimmed text
        trimmed_tokens = tokens[:end_idx]
        trimmed_text = "".join(trimmed_tokens).rstrip()
        
        processed_results.append(trimmed_text)
    
    return processed_results
