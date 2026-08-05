from __future__ import annotations

import os
import random
from collections import defaultdict
from importlib.resources import files

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

import jieba
from pypinyin import lazy_pinyin, Style
import math
import re
from typing import List

import sys
from pathlib import Path
import pyphen

SPEAKING_RATE_ROOT = Path(__file__).resolve().parents[2]

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


# tensor helpers

def lens_to_mask(t: int["b"], length: int | None = None) -> bool["b n"]:  # noqa: F722 F821
    if not exists(length):
        length = t.amax() # return the maximum value in tensor t

    seq = torch.arange(length, device=t.device)
    return seq[None, :] < t[:, None]

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

def count_syllables_pure(text: str) -> int:
    dic = pyphen.Pyphen(lang='en_US')
    total_syllables = 0
    
    # Define the regular expression
    #   [a-zA-Z']+  Matches one or more English letters or apostrophes (to handle words like "don't", "haven't")
    #   [\u4e00-\u9fff] Matches a single Chinese character
    pattern = re.compile(r"[a-zA-Z']+|[\u4e00-\u9fff]")
    tokens = pattern.findall(text)
    
    for token in tokens:
        if '\u4e00' <= token <= '\u9fff':
            total_syllables += 1
        else:
            try:
                syllables = dic.inserted(token.lower()).split("-")
                total_syllables += len(syllables)
            except Exception:
                total_syllables += 1
                
    return total_syllables

PUNCT_CHARS = set(',.?!;:。，、！？；：')
def count_punctuations(text):
    punct_syllables = 0
    for char in text:
        if char in PUNCT_CHARS:
            punct_syllables += 1
    return punct_syllables

def count_syllables(text: str) -> int:
    return count_syllables_pure(text) + count_punctuations(text)