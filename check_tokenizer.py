"""Verify the new Chinese-aware tokenizer."""
from tokenizers import Tokenizer
import os

# Load the v2 tokenizer
v2_path = "tokenizer_cache_v2/bpe_tokenizer.json"
t = Tokenizer.from_file(v2_path)
print(f"Vocab: {t.get_vocab_size()}")

tests = [
    "def hello():",
    "print('hello world')",
    "你好，今天天气怎么样？",
    "用户: 帮我写个程序\n助手: 好的！",
    "import numpy as np",
]

# Write results to file to avoid GBK terminal issues
with open("tokenizer_test.txt", "w", encoding="utf-8") as f:
    for text in tests:
        encoded = t.encode(text)
        decoded = t.decode(encoded.ids)
        f.write(f"Input: {text}\n")
        f.write(f"Tokens({len(encoded.tokens)}): {' | '.join(encoded.tokens[:20])}\n")
        f.write(f"IDs: {encoded.ids[:20]}\n")
        f.write(f"Decoded: {decoded[:80]}\n\n")

size_kb = os.path.getsize(v2_path) / 1024
print(f"Tokenizer saved ({size_kb:.0f} KB)")
print("Details written to tokenizer_test.txt")
