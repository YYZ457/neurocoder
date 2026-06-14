"""
NeuroCoder Data Pipeline
========================

Handles:
1. Downloading Python code datasets
2. Tokenization with trained BPE tokenizer (supports code + natural language)
3. Creating training batches for autoregressive LM
"""

import os
import re
import json
from typing import Optional, List, Dict, Tuple
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from tqdm import tqdm

# Use HuggingFace tokenizers for proper BPE (already installed with transformers)
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from tokenizers.normalizers import NFKC


# ===========================================================================
# BPE Tokenizer — trained on local data, handles code + natural language
# ===========================================================================

class CodeTokenizer:
    """
    Byte-level BPE tokenizer trained from local Python code.
    Handles: Python code, English text, comments, docstrings, any Unicode.
    """

    def __init__(
        self,
        vocab_size: int = 32768,
        cache_dir: str = "tokenizer_cache",
    ):
        self.vocab_size = vocab_size
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._tokenizer_path = os.path.join(cache_dir, "bpe_tokenizer.json")

        # Special token strings — must be defined before _create_tokenizer
        self.pad_token = "<|pad|>"
        self.bos_token = "<|bos|>"
        self.eos_token = "<|eos|>"

        # Try to load cached tokenizer
        if os.path.exists(self._tokenizer_path):
            print(f"  Loading cached BPE tokenizer: {self._tokenizer_path}")
            self._tokenizer = Tokenizer.from_file(self._tokenizer_path)
        else:
            print(f"  Creating new BPE tokenizer (will train on data)...")
            self._tokenizer = self._create_tokenizer()

        self.pad_token_id = self._tokenizer.token_to_id(self.pad_token) or 0
        self.bos_token_id = self._tokenizer.token_to_id(self.bos_token) or 1
        self.eos_token_id = self._tokenizer.token_to_id(self.eos_token) or 2

        self.vocab_size = self._tokenizer.get_vocab_size()
        print(f"  Tokenizer ready: vocab_size={self.vocab_size}")

    def _create_tokenizer(self) -> Tokenizer:
        """Create a fresh BPE tokenizer with no training (will be trained later)."""
        tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))

        # Byte-level pre-tokenizer handles all characters
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

        # Decoder
        tokenizer.decoder = decoders.ByteLevel()

        # Normalizer
        tokenizer.normalizer = NFKC()

        # Add special tokens
        special_tokens = ["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>"]
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=special_tokens,
            min_frequency=2,
            show_progress=True,
        )

        # Search for training data (project-root-relative or absolute)
        search_paths = [
            "sample_data/source",
            "chinese_data",
        ]
        train_files = []
        for sp in search_paths:
            p = Path(sp)
            if p.exists():
                train_files.extend(list(p.rglob("*.txt"))[:2000])
                train_files.extend(list(p.rglob("*.py"))[:2000])
        if train_files:
            train_files = [str(f) for f in train_files[:4000]]
            print(f"  Training BPE from {len(train_files)} files...")
            # Write consolidated corpus for speed
            corpus_path = os.path.join(self.cache_dir, "_train_corpus.txt")
            with open(corpus_path, "w", encoding="utf-8", errors="ignore") as out:
                for f_path in train_files:
                    try:
                        size = os.path.getsize(f_path)
                        if size < 100:
                            continue
                        # Sample more for large files
                        sample_size = min(size, 5_000_000)  # Up to 5MB per file for good vocab
                        with open(f_path, "r", encoding="utf-8", errors="ignore") as fh:
                            text = fh.read(sample_size)
                            if len(text) > 50:
                                out.write(text + "\n")
                    except:
                        pass
            tokenizer.train([corpus_path], trainer)
            os.remove(corpus_path)
            os.makedirs(self.cache_dir, exist_ok=True)
            tokenizer.save(self._tokenizer_path)
            print(f"  BPE tokenizer saved (vocab={tokenizer.get_vocab_size()})")
        else:
            # Need to pre-train with some text
            print(f"  No training files found, using sample text...")
            sample_texts = [
                "# Python code\n", "def function():\n    return 42\n",
                "class MyClass:\n    pass\n", "import os\nimport sys\n",
                "你好！\n用户: 你好\n助手: 你好！有什么可以帮助你的吗？\n",
            ]
            tokenizer.train_from_iterator(sample_texts, trainer)
            tokenizer.save(self._tokenizer_path)
            print(f"  Minimal tokenizer saved (vocab={tokenizer.get_vocab_size()})")

        # Post-processor: add BOS/EOS
        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{self.bos_token} $A {self.eos_token}",
            pair=f"{self.bos_token} $A {self.eos_token} {self.bos_token} $B {self.eos_token}",
            special_tokens=[
                (self.bos_token, tokenizer.token_to_id(self.bos_token) or 1),
                (self.eos_token, tokenizer.token_to_id(self.eos_token) or 2),
            ],
        )

        return tokenizer

    def train_on_data(self, file_paths: List[str]):
        """Re-train or fine-tune tokenizer on new files."""
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>"],
            min_frequency=2,
        )
        self._tokenizer.train(file_paths, trainer)
        self._tokenizer.save(self._tokenizer_path)
        self.vocab_size = self._tokenizer.get_vocab_size()
        self.pad_token_id = self._tokenizer.token_to_id(self.pad_token) or 0
        self.bos_token_id = self._tokenizer.token_to_id(self.bos_token) or 1
        self.eos_token_id = self._tokenizer.token_to_id(self.eos_token) or 2

    def encode(self, text: str, max_length: int = 2048) -> List[int]:
        """Encode text to token IDs."""
        encoded = self._tokenizer.encode(text)
        ids = encoded.ids
        # Truncate
        if len(ids) > max_length - 2:
            ids = ids[:max_length - 2]
        # Add BOS/EOS
        ids = [self.bos_token_id] + ids + [self.eos_token_id]
        return ids[:max_length]

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """Decode token IDs back to text."""
        special = {self.pad_token_id, self.bos_token_id, self.eos_token_id,
                   self._tokenizer.token_to_id("<|unk|>") or 3}
        if skip_special:
            ids = [i for i in ids if i not in special]
        return self._tokenizer.decode(ids)

    def __call__(self, texts: List[str], max_length: int = 2048) -> Dict[str, torch.Tensor]:
        """Batch encode."""
        all_ids = []
        all_masks = []
        for text in texts:
            ids = self.encode(text, max_length)
            L = len(ids)
            mask = [1] * L + [0] * (max_length - L)
            ids = ids + [self.pad_token_id] * (max_length - L)
            all_ids.append(ids[:max_length])
            all_masks.append(mask[:max_length])

        return {
            "input_ids": torch.tensor(all_ids, dtype=torch.long),
            "attention_mask": torch.tensor(all_masks, dtype=torch.long),
        }


class PythonCodeDataset(Dataset):
    """
    Dataset for Python code files.

    Can load from:
    1. A directory of .py files
    2. A JSONL file with {"code": "..."} entries
    3. The Stack dataset (via HuggingFace datasets)
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: CodeTokenizer,
        max_seq_len: int = 2048,
        cache_path: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.examples: List[List[int]] = []

        # Load data
        data_path = Path(data_path)
        if data_path.is_dir():
            self._load_from_directory(data_path)
        elif data_path.suffix == ".jsonl":
            self._load_from_jsonl(data_path)
        elif data_path.suffix == ".json":
            self._load_from_json(data_path)
        else:
            raise ValueError(f"Unsupported data format: {data_path}")

        # Cache tokenized data
        if cache_path:
            self._save_cache(cache_path)
        elif cache_path is None and len(self.examples) > 0:
            pass  # Don't cache automatically

        print(f"  Loaded {len(self.examples)} samples")

    def _tokenize_and_chunk(self, code: str) -> List[List[int]]:
        """Tokenize code and split into max_seq_len chunks."""
        token_ids = self.tokenizer.encode(code, max_length=10**6)  # No truncation
        chunks = []
        for i in range(0, len(token_ids), self.max_seq_len):
            chunk = token_ids[i:i + self.max_seq_len]
            if len(chunk) >= 32:  # Skip very short chunks
                # Pad if needed
                if len(chunk) < self.max_seq_len:
                    chunk = chunk + [self.tokenizer.pad_token_id] * (self.max_seq_len - len(chunk))
                chunks.append(chunk)
        return chunks

    def _get_text_from_item(self, item: dict) -> str:
        """Extract text from a JSON/dict item, trying common Chinese dataset fields."""
        for key in ["instruction", "input", "output", "q", "a", "question", "answer",
                     "content", "text", "sentence", "query", "response",
                     "code", "conversation", "messages", "chat", "reply"]:
            if key in item and isinstance(item[key], str) and len(item[key]) > 5:
                return item[key]
        # Multi-field: concatenate instruction + output
        if "instruction" in item and "output" in item:
            return f"{item['instruction']}\n{item['output']}"
        if "q" in item and "a" in item:
            return f"问: {item['q']}\n答: {item['a']}"
        if "question" in item and "answer" in item:
            return f"问: {item['question']}\n答: {item['answer']}"
        # Fallback: concat all string values
        texts = [str(v) for v in item.values() if isinstance(v, str) and len(v) > 5]
        return " ".join(texts) if texts else ""

    def _load_from_directory(self, path: Path):
        """Load all .py and .txt/.md files from a directory tree."""
        py_files = list(path.rglob("*.py"))
        txt_files = list(path.rglob("*.txt")) + list(path.rglob("*.md"))
        all_files = py_files + txt_files
        print(f"  Found {len(py_files)} .py + {len(txt_files)} .txt/.md in {path}")
        print(f"  Tokenizing {len(all_files)} files...")

        for code_file in tqdm(all_files, desc="  Tokenizing", unit="file"):
            try:
                size = code_file.stat().st_size
                # For large files (>100MB), stream in large chunks to avoid OOM
                if size > 100_000_000:
                    buffer = []
                    buf_len = 0
                    chunk_size = 10 * 1024 * 1024  # 10MB per chunk
                    for line in open(code_file, "r", encoding="utf-8", errors="ignore"):
                        buffer.append(line)
                        buf_len += len(line)
                        if buf_len >= chunk_size:
                            text = "".join(buffer)
                            chunks = self._tokenize_and_chunk(text)
                            self.examples.extend(chunks)
                            buffer = []
                            buf_len = 0
                    # Process remaining
                    if buffer:
                        text = "".join(buffer)
                        chunks = self._tokenize_and_chunk(text)
                        self.examples.extend(chunks)
                else:
                    code = code_file.read_text(encoding="utf-8")
                    chunks = self._tokenize_and_chunk(code)
                    self.examples.extend(chunks)
            except Exception as e:
                pass  # Skip unreadable files silently

    def _load_from_jsonl(self, path: Path):
        """Load from JSONL with Chinese dataset field names."""
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    if isinstance(item, dict):
                        code = self._get_text_from_item(item)
                    elif isinstance(item, str):
                        code = item
                    else:
                        continue
                    if len(code) > 10:
                        chunks = self._tokenize_and_chunk(code)
                        self.examples.extend(chunks)
                except Exception:
                    continue

    def _load_from_json(self, path: Path):
        """Load from JSON file (list of strings or objects)."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [data]
        for item in data:
            if isinstance(item, str):
                code = item
            elif isinstance(item, dict):
                code = self._get_text_from_item(item)
            else:
                continue
            chunks = self._tokenize_and_chunk(code)
            self.examples.extend(chunks)

    def _save_cache(self, cache_path: str):
        """Save tokenized data to disk."""
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save({
            "examples": self.examples,
            "vocab_size": self.tokenizer.vocab_size,
            "pad_token_id": self.tokenizer.pad_token_id,
        }, cache_path)
        print(f"  Cached to {cache_path}")

    @classmethod
    def from_cache(cls, cache_path: str, max_seq_len: int = 2048):
        """Load pre-tokenized dataset from cache."""
        data = torch.load(cache_path, weights_only=False)
        dataset = cls.__new__(cls)
        dataset.examples = data["examples"]
        dataset.max_seq_len = max_seq_len
        dataset.pad_token_id = data.get("pad_token_id", 0)
        # Truncate to max_seq_len if needed
        dataset.examples = [ex[:max_seq_len] for ex in dataset.examples]
        return dataset

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        token_ids = self.examples[idx][:self.max_seq_len]
        pad_id = getattr(self, 'pad_token_id', 0)

        # Pad
        if len(token_ids) < self.max_seq_len:
            token_ids = token_ids + [pad_id] * (self.max_seq_len - len(token_ids))

        input_ids = torch.tensor(token_ids, dtype=torch.long)
        labels = input_ids.clone()
        attention_mask = (input_ids != pad_id).long()

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


class StreamingCodeDataset(IterableDataset):
    """
    Streaming dataset for large code corpora.
    Reads files on-the-fly, never loads everything into memory.
    Useful for training on The Stack or other massive datasets.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: CodeTokenizer,
        max_seq_len: int = 2048,
        buffer_size: int = 10000,
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.buffer_size = buffer_size

    def _iter_py_files(self):
        """Yield file paths."""
        if self.data_path.is_dir():
            yield from self.data_path.rglob("*.py")
        elif self.data_path.suffix in (".jsonl", ".json"):
            yield self.data_path

    def __iter__(self):
        buffer = []
        for file_path in self._iter_py_files():
            try:
                if file_path.suffix == ".py":
                    code = file_path.read_text(encoding="utf-8")
                elif file_path.suffix == ".jsonl":
                    with open(file_path, "r", encoding="utf-8") as f:
                        for line in f:
                            item = json.loads(line)
                            code = item.get("code") or item.get("content", "")
                            token_ids = self.tokenizer.encode(code, 10**6)
                            buffer.extend(self._chunk(token_ids))
                            while len(buffer) >= self.buffer_size:
                                chunk = buffer.pop(0)
                                yield self._to_tensors(chunk)
                    continue  # JSONL processed inline, skip to next file
                else:
                    continue  # Skip unsupported file types
            except Exception:
                continue

            token_ids = self.tokenizer.encode(code, 10**6)
            buffer.extend(self._chunk(token_ids))
            while len(buffer) >= self.buffer_size:
                chunk = buffer.pop(0)
                yield self._to_tensors(chunk)

        # Yield remaining
        for chunk in buffer:
            yield self._to_tensors(chunk)

    def _chunk(self, token_ids: List[int]) -> List[List[int]]:
        chunks = []
        for i in range(0, len(token_ids), self.max_seq_len):
            chunk = token_ids[i:i + self.max_seq_len]
            if len(chunk) >= 32:
                if len(chunk) < self.max_seq_len:
                    chunk = chunk + [self.tokenizer.pad_token_id] * (self.max_seq_len - len(chunk))
                chunks.append(chunk)
        return chunks

    def _to_tensors(self, token_ids: List[int]) -> Dict[str, torch.Tensor]:
        input_ids = torch.tensor(token_ids[:self.max_seq_len], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
            "attention_mask": (input_ids != self.tokenizer.pad_token_id).long(),
        }


def create_dataloaders(
    train_path: str,
    tokenizer: CodeTokenizer,
    config,  # NeuroCoderConfig
    eval_path: Optional[str] = None,
    cache_dir: str = "data_cache",
    num_workers: int = 2,
) -> tuple:
    """
    Create training and evaluation dataloaders.

    Args:
        train_path: Path to training data (directory or JSONL)
        tokenizer: CodeTokenizer instance
        config: NeuroCoderConfig
        eval_path: Optional eval data path
        cache_dir: Directory for caching tokenized data
        num_workers: DataLoader workers

    Returns:
        (train_loader, eval_loader) — eval_loader may be None
    """
    max_seq_len = config.max_seq_len

    # Create cache path
    os.makedirs(cache_dir, exist_ok=True)
    train_cache = os.path.join(cache_dir, "train_cache.pt")

    # Try to load from cache
    if os.path.exists(train_cache):
        print("Loading cached training data...")
        train_dataset = PythonCodeDataset.from_cache(train_cache, max_seq_len)
    else:
        print(f"Building training dataset from {train_path}...")
        train_dataset = PythonCodeDataset(
            train_path, tokenizer, max_seq_len, cache_path=train_cache
        )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    eval_loader = None
    if eval_path:
        eval_cache = os.path.join(cache_dir, "eval_cache.pt")
        if os.path.exists(eval_cache):
            print("Loading cached eval data...")
            eval_dataset = PythonCodeDataset.from_cache(eval_cache, max_seq_len)
        else:
            print(f"Building eval dataset from {eval_path}...")
            eval_dataset = PythonCodeDataset(
                eval_path, tokenizer, max_seq_len, cache_path=eval_cache
            )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_loader, eval_loader


def download_sample_data(output_dir: str = "sample_data"):
    """
    Download some sample Python code for initial testing.
    Uses the Python standard library as a small, high-quality dataset.
    """
    import subprocess
    import sys

    os.makedirs(output_dir, exist_ok=True)

    print("Downloading sample Python code...")
    print("  Using popular Python packages from PyPI...")

    packages = [
        "requests", "flask", "numpy", "pandas", "rich",
        "click", "pydantic", "fastapi", "aiohttp", "black",
    ]

    for pkg in packages:
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "download", pkg, "--no-deps", "-d", output_dir],
                capture_output=True, timeout=30,
            )
            print(f"  [OK] Downloaded {pkg}")
        except Exception as e:
            print(f"  [FAIL] Failed: {pkg} ({e})")

    # Extract source from wheels
    print("\n  Extracting source code...")
    import zipfile, tarfile, io
    source_dir = os.path.join(output_dir, "source")
    os.makedirs(source_dir, exist_ok=True)

    for f in os.listdir(output_dir):
        fpath = os.path.join(output_dir, f)
        if f.endswith(".whl"):
            try:
                with zipfile.ZipFile(fpath) as zf:
                    for name in zf.namelist():
                        if name.endswith(".py"):
                            zf.extract(name, source_dir)
                print(f"  [OK] Extracted {f}")
            except Exception:
                pass
        elif f.endswith(".tar.gz"):
            try:
                with tarfile.open(fpath) as tf:
                    for member in tf.getmembers():
                        if member.name.endswith(".py"):
                            tf.extract(member, source_dir)
                print(f"  [OK] Extracted {f}")
            except Exception:
                pass

    print(f"\n  Sample data ready at: {source_dir}")
    return source_dir


if __name__ == "__main__":
    # Quick test
    print("Testing CodeTokenizer...")
    tokenizer = CodeTokenizer()

    code = '''def fibonacci(n: int) -> int:
    """Return the nth Fibonacci number."""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b

class DataLoader:
    def __init__(self, data, batch_size=32):
        self.data = data
        self.batch_size = batch_size
'''

    encoded = tokenizer.encode(code)
    print(f"  Encoded {len(code)} chars → {len(encoded)} tokens")
    print(f"  Compression ratio: {len(code)/len(encoded):.1f}x")

    decoded = tokenizer.decode(encoded)
    print(f"  Decoded: {decoded[:100]}...")

    print("\n  [OK] Tokenizer is working!")
