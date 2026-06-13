#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NeuroCoder Cloud — downloads REAL Chinese data. Only verified repos."""
import os, subprocess, urllib.request, json, re, random, shutil
from pathlib import Path

os.environ["GIT_TERMINAL_PROMPT"] = "0"
GIT = ["git", "-c", "http.sslVerify=false"]
DATA_DIR = Path("chinese_data")
DATA_DIR.mkdir(exist_ok=True, parents=True)

def clone(repo, dest_dir=""):
    name = repo.split("/")[-1]
    dest = Path(dest_dir or DATA_DIR) / name
    if dest.exists(): return
    try:
        subprocess.run(GIT + ["clone", "--depth", "1", f"https://github.com/{repo}.git", str(dest)],
                      capture_output=True, timeout=180, check=True)
        print(f"  [OK] {name}")
    except Exception as e:
        print(f"  [FAIL] {name}")

def dl(url, dest, resume=True):
    if dest.exists() and not resume: return
    if dest.exists() and dest.stat().st_size > 1024: print(f"  [Skip] {dest.name}"); return
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    try:
        print(f"  Downloading {dest.name}...", end=" ", flush=True)
        urllib.request.urlretrieve(url, dest)
        print(f"[{dest.stat().st_size/1024/1024:.0f} MB]")
    except Exception as e:
        print(f"[FAIL] {e}")

def convert_jsonl_to_txt():
    """Convert all JSON/JSONL to .txt for the pipeline."""
    conv = DATA_DIR / "_txt"
    conv.mkdir(exist_ok=True)
    for f in list(DATA_DIR.rglob("*.jsonl")) + list(DATA_DIR.rglob("*.json")):
        if "node_modules" in str(f) or "package.json" in str(f): continue
        out = conv / f"{f.parent.name}_{f.stem}.txt"
        if out.exists(): continue
        count = 0
        try:
            with open(out, "w", encoding="utf-8") as fo:
                for line in open(f, encoding="utf-8", errors="ignore"):
                    try:
                        item = json.loads(line)
                        texts = []
                        for v in item.values():
                            if isinstance(v, str) and len(v) > 5:
                                texts.append(v)
                        if texts:
                            fo.write(" ".join(texts) + "\n")
                            count += 1
                    except: pass
            if count == 0: out.unlink()  # Remove empty files
        except: pass

print("=" * 60)
print("  Downloading REAL Chinese data (verified repos only)")
print("=" * 60)

# =============================================================
# 1. LARGEST sources (verified, 1GB+ each)
# =============================================================
print("\n[1/5] Large Chinese corpora...")

# nlp_chinese_corpus — contains Wikipedia, news, Weibo (~1GB+)
clone("brightmart/nlp_chinese_corpus")

# CLUECorpus2020 — large news dataset
clone("CLUEbenchmark/CLUECorpus2020")

# Chinese NLP collection
clone("SophonPlus/ChineseNlpCorpus")
clone("InsaneLife/ChineseNLPCorpus")

# =============================================================
# 2. Chinese tech & education (verified)
# =============================================================
print("\n[2/5] Chinese tech & education...")

clone("jackfrued/Python-100-Days")      # 100 days Python in Chinese
clone("d2l-ai/d2l-zh")                 # Dive into Deep Learning
clone("xitu/gold-miner")               # Chinese tech translations
clone("xitu/tensorflow-docs")          # TF docs in Chinese
clone("CyC2018/CS-Notes")              # CS notes in Chinese
clone("MLEveryday/100-Days-Of-ML-Code")
clone("scutan90/DeepLearning-500-questions")

# Large Chinese AI frameworks (have Chinese comments/docstrings)
clone("PaddlePaddle/Paddle", DATA_DIR / "code")
clone("PaddlePaddle/PaddleNLP", DATA_DIR / "code")
clone("PaddlePaddle/PaddleOCR", DATA_DIR / "code")
clone("PaddlePaddle/Book", DATA_DIR / "code")

# =============================================================
# 3. Chinese text + NLP libraries (verified)
# =============================================================
print("\n[3/5] Chinese text & NLP...")

clone("chinese-poetry/chinese-poetry")  # Chinese poetry

# jieba — has Chinese dictionary text data
clone("fxsjy/jieba")

# THU/CoAI dialogue
clone("thu-coai/CDial-GPT")

# CN code generation model (has Chinese docstrings)
clone("THUDM/CodeGeeX")

# Microsoft UNILM (has Chinese pretraining data references)
clone("microsoft/unilm")

# =============================================================
# 4. Python code (verified)
# =============================================================
print("\n[4/5] Python code...")
py_dir = DATA_DIR / "python"
py_dir.mkdir(exist_ok=True)

for repo in ["pallets/flask", "psf/black", "pydantic/pydantic",
              "Textualize/rich", "kennethreitz/requests",
              "pallets/click", "pytest-dev/pytest"]:
    clone(repo, py_dir)

# =============================================================
# 5. Wikipedia via wget with resume
# =============================================================
print("\n[5/5] Wikipedia (via wget)...")
wiki_dir = DATA_DIR / "wiki"
wiki_dir.mkdir(exist_ok=True)
wiki_bz2 = wiki_dir / "zhwiki-latest-pages-articles.xml.bz2"

if not wiki_bz2.exists():
    url = "https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2"
    try:
        subprocess.run(["wget", "-c", "-O", str(wiki_bz2), url],
                      timeout=7200, check=True)
        print(f"  [OK] {wiki_bz2.stat().st_size/1024/1024/1024:.1f} GB")
    except Exception as e:
        print(f"  [FAIL] wget: {e}")

# Extract Wikipedia XML → plain text
def extract_wiki():
    import bz2, xml.sax
    out_path = wiki_dir / "wiki.txt"
    if out_path.exists(): return

    class Handler(xml.sax.ContentHandler):
        def __init__(self, out):
            self.out = out; self.text = ""; self.title = ""
            self.in_text = False; self.in_title = False; self.count = 0
        def startElement(self, name, attrs):
            if name == "text": self.in_text = True
            if name == "title": self.in_title = True
        def endElement(self, name):
            if name == "text":
                if len(self.text) > 100:
                    cleaned = re.sub(r'[\[\]\{\}\'\*#\|-]', ' ', self.text)
                    cleaned = re.sub(r'\n+', '\n', cleaned).strip()
                    if len(cleaned) > 100:
                        self.out.write(cleaned + "\n\n"); self.count += 1
                self.text = ""; self.in_text = False
            if name == "title": self.title = ""; self.in_title = False
        def characters(self, content):
            if self.in_text: self.text += content
            if self.in_title: self.title += content

    if wiki_bz2.exists() and wiki_bz2.stat().st_size > 1e8:
        print("  Extracting Wikipedia text (~20 min)...")
        with bz2.open(wiki_bz2, "rb") as f:
            with open(out_path, "w", encoding="utf-8") as out:
                xml.sax.parse(f, Handler(out))
        print(f"  [OK] Wikipedia text ready")

extract_wiki()

# =============================================================
# Convert JSON/JSONL to .txt
# =============================================================
print("\nConverting JSON -> .txt...")
convert_jsonl_to_txt()

# =============================================================
# Retrain Tokenizer
# =============================================================
print("\nRetraining tokenizer with Chinese...")
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, normalizers

train_files = []
for ext in ["*.txt", "*.py", "*.md"]:
    train_files.extend(DATA_DIR.rglob(ext))
random.seed(42)
sample = random.sample(train_files, min(len(train_files), 50000))

corpus_path = "_train_corpus.txt"
with open(corpus_path, "w", encoding="utf-8", errors="ignore") as out:
    for f in sample:
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
            if len(text) > 50: out.write(text[:5000] + "\n")
        except: pass

tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()
tokenizer.normalizer = normalizers.NFKC()
trainer = trainers.BpeTrainer(vocab_size=32768,
    special_tokens=["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>"],
    min_frequency=2, show_progress=True)
tokenizer.train([corpus_path], trainer)
os.makedirs("tokenizer_cache", exist_ok=True)
tokenizer.save("tokenizer_cache/bpe_tokenizer.json")
os.remove(corpus_path)

# =============================================================
# Summary
# =============================================================
text_files = list(DATA_DIR.rglob("*.txt")) + list(DATA_DIR.rglob("*.md")) + list(DATA_DIR.rglob("*.py"))
text_files = [f for f in text_files if f.is_file()]
total_bytes = sum(f.stat().st_size for f in text_files)
wiki_size = (wiki_dir/"wiki.txt").stat().st_size if (wiki_dir/"wiki.txt").exists() else 0

print(f"""
{'='*60}
  DONE
{'='*60}
  Wiki text:     {wiki_size/1024/1024/1024:.1f} GB  (if download succeeded)
  Other data:    {(total_bytes-wiki_size)/1024/1024:.0f} MB
  Total:         {total_bytes/1024/1024/1024:.1f} GB
  Text files:    {len(text_files):,}

  TRAIN:
  rm -rf data_cache
  python train.py --config small --steps 200000 --data chinese_data
""")
