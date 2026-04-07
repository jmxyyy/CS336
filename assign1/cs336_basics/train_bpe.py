import os
from collections import Counter, defaultdict
from functools import total_ordering
import heapq
from multiprocessing import Pool
from pathlib import Path

from cs336_basics.common import PAT_RE, BYTE_TABLE, build_special_token_pattern, merge_once, save_tokenizer_artifacts
from cs336_basics.pretokenization_example import find_chunk_boundaries


def _count_pretokens_in_chunk(args) -> Counter[tuple[bytes, ...]]:
    """统计分块内预分词频率

    Args:
        args ([TODO:parameter]): worker参数
        input_path, start, end, special_tokens = args

    Returns:
        块内预分词频率表
    """
    input_path, start, end, special_tokens = args
    split_re = build_special_token_pattern(special_tokens)
    counts = Counter()

    # 读取文件分块
    with open(input_path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")

    # special_tokens正则块内分段
    segments = split_re.split(text) if split_re is not None else [text]

    for seg in segments:
        # 分段进行PAT正则
        for match in PAT_RE.finditer(seg):
            token_bytes = match.group(0).encode("utf-8")
            # 字节串拆分为单字节序列
            token_tuple = tuple(BYTE_TABLE[b] for b in token_bytes)
            counts[token_tuple] += 1

    return counts


def _build_pretoken_counts_parallel(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    num_processes: int | None = None,
) -> Counter[tuple[bytes, ...]]:
    """并行统计语料预分词频率

    Args:
        input_path: [TODO:description]
        special_tokens: [TODO:description]
        num_processes: [TODO:description]

    Returns:
        整份语料文件预分词后频率表
    """
    path = Path(input_path)

    # 无special_tokens使用单线程统计
    if not special_tokens:
        file_size = path.stat().st_size
        return _count_pretokens_in_chunk((str(path), 0, file_size, tuple()))

    if num_processes is None:
        num_processes = os.cpu_count() or 1

    # 取special_tokens作为分块边界标记
    boundary_token = special_tokens[0].encode("utf-8")

    with path.open("rb") as f:
        # 分块
        boundaries = find_chunk_boundaries(f, num_processes, boundary_token)

    # 构造workers
    workers = [
        (str(path), start, end, tuple(special_tokens))
        for start, end in zip(boundaries[:-1], boundaries[1:])
        if start < end
    ]

    if len(workers) <= 1:
        return _count_pretokens_in_chunk(workers[0])

    # 线程池
    with Pool(processes=min(num_processes, len(workers))) as pool:
        partial_counts = pool.map(_count_pretokens_in_chunk, workers)

    # 汇总频率表
    total_counts = Counter()
    for counts in partial_counts:
        total_counts.update(counts)
    return total_counts


def _pair_iter(token: tuple[bytes, ...]):
    """pair迭代器

    Args:
        token: [TODO:description]

    Yields:
        [TODO:description]
    """
    for i in range(len(token) - 1):
        yield (token[i], token[i + 1])


def _build_pair_index(
    token_frequency_table: dict[tuple[bytes, ...], int],
) -> tuple[Counter[tuple[bytes, bytes]], dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]]:
    """利用频率表构造pair索引

    Args:
        token_frequency_table: [TODO:description]

    Returns:
        [TODO:return]
    """
    # 统计每对pair频率
    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    # 统计每对pair出现在哪些token序列
    pair2tokens: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(set)

    for token, freq in token_frequency_table.items():
        # 集合去重
        seen_pairs = set()
        for pair in _pair_iter(token):
            pair_counts[pair] += freq
            seen_pairs.add(pair)
        for pair in seen_pairs:
            pair2tokens[pair].add(token)

    return pair_counts, pair2tokens


@total_ordering
class _MaxPairKey:
    __slots__ = ("pair",)  # 节省对象开销

    def __init__(self, pair: tuple[bytes, bytes]):
        self.pair = pair

    def __lt__(self, other: "_MaxPairKey") -> bool:
        """维护最小堆,因此反转比较方向来返回实际字典序更大的pair

        Args:
            other: [TODO:description]

        Returns:
            [TODO:return]
        """
        return self.pair > other.pair

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _MaxPairKey) and self.pair == other.pair


def _build_pair_heap(pair_counts: Counter[tuple[bytes, bytes]]):
    """建堆

    Args:
        pair_counts: [TODO:description]

    Returns:
        [TODO:return]
    """
    heap = [(-count, _MaxPairKey(pair), pair) for pair, count in pair_counts.items() if count > 0]
    heapq.heapify(heap)
    return heap


def _push_pair_heap(heap, pair_counts: Counter[tuple[bytes, bytes]], pair: tuple[bytes, bytes]) -> None:
    count = pair_counts.get(pair, 0)
    if count > 0:
        heapq.heappush(heap, (-count, _MaxPairKey(pair), pair))


def _pop_best_pair(heap, pair_counts: Counter[tuple[bytes, bytes]]) -> tuple[bytes, bytes]:
    while heap:
        neg_count, _, pair = heapq.heappop(heap)
        live_count = pair_counts.get(pair, 0)
        # 检验
        if live_count > 0 and live_count == -neg_count:
            return pair
    raise RuntimeError("No valid pair in heap")


def _remove_token_from_index(
    token: tuple[bytes, ...],
    freq: int,
    pair_counts: Counter[tuple[bytes, bytes]],
    pair2tokens: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]],
) -> set[tuple[bytes, bytes]]:
    changed_pairs: set[tuple[bytes, bytes]] = set()
    for pair in _pair_iter(token):
        new_count = pair_counts[pair] - freq
        if new_count > 0:
            pair_counts[pair] = new_count
        else:
            del pair_counts[pair]
        changed_pairs.add(pair)

    for pair in changed_pairs:
        bucket = pair2tokens[pair]
        bucket.discard(token)
        if not bucket:
            del pair2tokens[pair]

    return changed_pairs


def _add_token_into_index(
    token: tuple[bytes, ...],
    freq: int,
    pair_counts: Counter[tuple[bytes, bytes]],
    pair2tokens: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]],
) -> set[tuple[bytes, bytes]]:
    changed_pairs: set[tuple[bytes, bytes]] = set()
    for pair in _pair_iter(token):
        pair_counts[pair] += freq
        changed_pairs.add(pair)
    for pair in changed_pairs:
        pair2tokens[pair].add(token)
    return changed_pairs


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

    token_frequency_table = _build_pretoken_counts_parallel(
        input_path=input_path,
        special_tokens=special_tokens,
        num_processes=kwargs.get("num_processes"),
    )

    # 存储合并记录
    merges: list[tuple[bytes, bytes]] = []

    pair_counts, pair2tokens = _build_pair_index(token_frequency_table)
    pair_heap = _build_pair_heap(pair_counts)

    while len(vocab) < vocab_size:
        if not pair_counts:
            break

        best_pair = _pop_best_pair(pair_heap, pair_counts)

        # 记录合并
        merges.append(best_pair)

        # 新token加入词汇表
        vocab[vacab_index] = best_pair[0] + best_pair[1]
        vacab_index += 1

        affected_tokens = list(pair2tokens.get(best_pair, ()))

        changed_pairs: set[tuple[bytes, bytes]] = set()

        for token in affected_tokens:
            freq = token_frequency_table.pop(token)
            changed_pairs |= _remove_token_from_index(token, freq, pair_counts, pair2tokens)
            new_token_frequency_seq = merge_once(token, best_pair)
            token_frequency_table[new_token_frequency_seq] += freq
            changed_pairs |= _add_token_into_index(new_token_frequency_seq, freq, pair_counts, pair2tokens)
        for pair in changed_pairs:
            _push_pair_heap(pair_heap, pair_counts, pair)

    return vocab, merges


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    default_input_path = project_root / "data" / "TinyStoriesV2-GPT4-valid.txt"
    # default_input_path = project_root / "tests" / "fixtures" / "tinystories_sample_5M.txt"
    vocab, merges = train_bpe(default_input_path, 32000, ["<|endoftext|>"])
    save_tokenizer_artifacts(vocab, merges, "vocab.json", "merges.txt")
