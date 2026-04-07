from __future__ import annotations

import json
import os
from pathlib import Path
import regex

from tests.common import gpt2_bytes_to_unicode

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
# PAT分词规则预编译
PAT_RE = regex.compile(PAT)
# 0..255到bytes映射表
BYTE_TABLE = tuple(bytes([i]) for i in range(256))
BYTES_TO_UNICODE = gpt2_bytes_to_unicode()
UNICODE_TO_BYTES = {v: bytes([k]) for k, v in BYTES_TO_UNICODE.items()}


def pretokenize(text: str) -> list[str]:
    """PAT预分词

    Args:
        text: [TODO:description]

    Returns:
        [TODO:return]
    """
    return PAT_RE.findall(text)


def build_special_token_pattern(
    special_tokens: list[str] | tuple[str, ...] | None,
    *,
    capture: bool = False,
) -> regex.Pattern | None:
    """构造special_token正则

    Args:
        special_tokens: [TODO:description]
        capture: [TODO:description]

    Returns:
        [TODO:return]
    """
    if not special_tokens:
        return None

    escaped = [regex.escape(token) for token in sorted(special_tokens, key=len, reverse=True)]
    body = "|".join(escaped)
    if capture:
        body = f"({body})"
    return regex.compile(body)


def merge_once(
    token_seq: tuple[bytes, ...],
    pair: tuple[bytes, bytes],
) -> tuple[bytes, ...]:
    """BPE 合并token对

    Args:
        token_seq: [TODO:description]
        pair: [TODO:description]

    Returns:
        [TODO:return]
    """
    merged_token = pair[0] + pair[1]
    out: list[bytes] = []
    i = 0
    while i < len(token_seq):
        if i + 1 < len(token_seq) and (token_seq[i], token_seq[i + 1]) == pair:
            out.append(merged_token)
            i += 2
        else:
            out.append(token_seq[i])
            i += 1
    return tuple(out)


def bytes_to_serialized_token(token: bytes) -> str:
    return "".join(BYTES_TO_UNICODE[b] for b in token)


def serialized_token_to_bytes(token: str) -> bytes:
    return b"".join(UNICODE_TO_BYTES[ch] for ch in token)


def save_tokenizer_artifacts(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    vocab_filepath: str | os.PathLike,
    merges_filepath: str | os.PathLike,
) -> None:
    serial_vocab = {bytes_to_serialized_token(token_bytes): token_id for token_id, token_bytes in vocab.items()}
    with open(vocab_filepath, "w", encoding="utf-8") as f:
        json.dump(serial_vocab, f, ensure_ascii=False)

    with open(merges_filepath, "w", encoding="utf-8") as f:
        for left, right in merges:
            f.write(f"{bytes_to_serialized_token(left)} {bytes_to_serialized_token(right)}\n")


def load_tokenizer_artifacts(
    vocab_filepath: str | os.PathLike,
    merges_filepath: str | os.PathLike,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(vocab_filepath, encoding="utf-8") as f:
        raw_vocab = json.load(f)
    vocab = {token_id: serialized_token_to_bytes(token_str) for token_str, token_id in raw_vocab.items()}

    merges: list[tuple[bytes, bytes]] = []
    with open(merges_filepath, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            left_str, right_str = line.split(" ")
            merges.append(
                (
                    serialized_token_to_bytes(left_str),
                    serialized_token_to_bytes(right_str),
                )
            )
    return vocab, merges
