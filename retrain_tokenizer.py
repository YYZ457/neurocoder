"""Retrain BPE tokenizer with Python code + Chinese text included."""
import sys; sys.path.insert(0,'.')
import os
from pathlib import Path
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, normalizers

DATA_DIR = "sample_data/source"
CACHE_DIR = "tokenizer_cache_v2"
os.makedirs(CACHE_DIR, exist_ok=True)

print("Collecting training files...")
all_files = []
for ext in ["*.py", "*.txt", "*.md"]:
    all_files.extend(list(Path(DATA_DIR).rglob(ext)))

# Include Chinese data specifically
chinese_dir = Path(DATA_DIR) / "_chinese_data"
if chinese_dir.exists():
    all_files.extend(list(chinese_dir.rglob("*.txt")))

# Filter to text files (readable)
text_files = [f for f in all_files if f.suffix in ['.py', '.txt', '.md']]
print(f"Found {len(text_files)} potential training files")

# Take a representative sample - include ALL Chinese files, then add .py files
chinese_files = [f for f in text_files if f.suffix == '.txt' or 'chinese' in str(f).lower() or 'cn' in str(f.stem).lower()]
py_files = [f for f in text_files if f.suffix == '.py']

print(f"  Chinese-related files: {len(chinese_files)}")
print(f"  Python files: {len(py_files)}")

# Balance: use all Chinese files + up to 8000 .py files
sample_files = chinese_files + py_files[:8000]
print(f"Training on {len(sample_files)} files total")

# Create tokenizer
tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()
tokenizer.normalizer = normalizers.NFKC()

special_tokens = ["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>"]
trainer = trainers.BpeTrainer(
    vocab_size=32768,
    special_tokens=special_tokens,
    min_frequency=2,
    show_progress=True,
)

# Train - write consolidated clean file first to avoid encoding issues
print("Preparing clean training corpus...")
clean_path = os.path.join(CACHE_DIR, "train_corpus.txt")
with open(clean_path, 'w', encoding='utf-8', errors='ignore') as out:
    for f in sample_files:
        try:
            text = f.read_text(encoding='utf-8', errors='ignore')
            if len(text) > 20:  # Skip tiny files
                out.write(text)
                out.write('\n')
        except:
            pass

print(f"Consolidated corpus size: {os.path.getsize(clean_path)/1024/1024:.1f} MB")
print("Training BPE tokenizer with code + Chinese...")
tokenizer.train([clean_path], trainer)

# Save
path = os.path.join(CACHE_DIR, "bpe_tokenizer.json")
tokenizer.save(path)

# Verify
print(f"\nTokenizer saved to {path}")
print(f"Vocab size: {tokenizer.get_vocab_size()}")

# Test Chinese encoding
test_texts = [
    "def hello():",
    "print('hello world')",
    "你好，今天天气怎么样？",
    "用户: 帮我写个程序\n助手: 好的！",
    "import numpy as np",
    "def fibonacci(n):  # 计算斐波那契数列",
]

for text in test_texts:
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded.ids)
    print(f"\n  Input: {text}")
    print(f"  Tokens: {encoded.tokens[:15]}...")
    print(f"  IDs: {encoded.ids[:15]}...")
    print(f"  Decoded: {decoded[:60]}...")
