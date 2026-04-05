import os
from collections import defaultdict
from pathlib import Path
import pickle
import regex

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def merge_token_sequence(token_seq: tuple, best_pair: tuple, new_token: bytes) -> tuple:
    """合并最佳对

    Args:
        token_seq: [TODO:description]
        best_pair: [TODO:description]
        new_token: [TODO:description]

    Returns:
        [TODO:return]
    """
    new_seq = []
    i = 0
    while i < len(token_seq):
        if i < len(token_seq) - 1 and (token_seq[i], token_seq[i + 1]) == best_pair:
            new_seq.append(new_token)
            i += 2
        else:
            new_seq.append(token_seq[i])
            i += 1
    return tuple(new_seq)


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """

    # # 第0步要先校验一下参数，为了更好地增强函数的鲁棒性
    # if not isinstance(vocab_size, int) or vocab_size <= 0:
    #     raise ValueError("vocab_size 必须是一个正整数。")

    # 词表初始化
    # 字节级BPE Tokenizer： 初始词表为单字节集合，共256可能
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    vacab_index: int = 256

    # 频率表
    token_frequency_table = defaultdict(int)

    # 用集合检查特殊符号的字节表示是否已存在于词汇表中
    check_set: set[bytes] = set(vocab.values())

    # 特殊符号加入词汇表
    for st in special_tokens:
        if len(vocab) >= vocab_size:
            break
        st_bytes: bytes = st.encode("utf-8")
        if st_bytes not in check_set:
            vocab[vacab_index] = st_bytes
            check_set.add(st_bytes)
            vacab_index += 1

    # 加载语料库
    corpus_path = Path(input_path)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Training corpus not found: {corpus_path.resolve()}")

    with corpus_path.open(encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # 预分词
    # special_tokens分割
    chunks = regex.split("|".join(map(regex.escape, special_tokens)), text)
    # PAT再分割
    for chunk in chunks:
        for word in regex.findall(PAT, chunk):
            word_bytes = word.encode("utf-8")
            bytes_list = [bytes([x]) for x in word_bytes]
            token_frequency_table[tuple(bytes_list)] += 1

    # 存储合并记录
    merges: list[tuple[bytes, bytes]] = []

    # 统计所有token对频率
    pair_counts = defaultdict(int)
    for token in token_frequency_table.keys():
        for i in range(len(token) - 1):
            pair_counts[token[i], token[i + 1]] += token_frequency_table[token]

    # 训练BPE
    while len(vocab) < vocab_size:
        if not pair_counts:
            break

        # 最大频率
        max_freq = max(pair_counts.values())

        # 频率最高的候选token对
        candidates = [k for k, v in pair_counts.items() if v == max_freq]

        # 选择字节序最大的
        best_pair = max(candidates)

        # 记录合并
        merges.append(best_pair)

        # 连接token
        new_token_bytes = best_pair[0] + best_pair[1]

        # 新token加入词汇表
        vocab[vacab_index] = new_token_bytes
        vacab_index += 1

        # 记录受影响的token，也就是包含best_pair的来自token_frequency_table的token
        affected_tokens = []
        for token, freq in token_frequency_table.items():
            has_pair = any(token[i : i + 2] == best_pair for i in range(len(token) - 1))
            if has_pair:
                affected_tokens.append((token, freq))

        # 从受影响的token中出发,每个token就是token_frequency_table的key
        for token, freq in affected_tokens:
            # 删除pair_counts中对应的best_pair
            for i in range(len(token) - 1):
                pair_counts[token[i], token[i + 1]] -= freq
                if pair_counts[token[i], token[i + 1]] <= 0:
                    del pair_counts[token[i], token[i + 1]]

            # 将best_pair合并为new_token
            new_token_frequency_seq = merge_token_sequence(token, best_pair, new_token_bytes)

            # 更新pair_counts
            for i in range(len(new_token_frequency_seq) - 1):
                pair = (new_token_frequency_seq[i], new_token_frequency_seq[i + 1])
                pair_counts[pair] += freq

            # 更新token_frequency_table
            del token_frequency_table[token]
            token_frequency_table[new_token_frequency_seq] += freq

    # 保存文件
    with open("vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open("merges.pkl", "wb") as f:
        pickle.dump(merges, f)

    return vocab, merges


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    default_input_path = project_root / "data" / "owt_valid.txt"
    vocab, merges = train_bpe(default_input_path, 20000, [""])
    # print(vocab)
    # print(merges)
