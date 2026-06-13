#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuroCoder Cloud — 5GB+ REAL Chinese data from GitHub + Wikipedia.
"""
import os, subprocess, urllib.request, gzip, json, random, re, sys
from pathlib import Path

os.environ["GIT_TERMINAL_PROMPT"] = "0"
GIT = ["git", "-c", "http.sslVerify=false"]
DATA_DIR = Path("chinese_data")
DATA_DIR.mkdir(exist_ok=True, parents=True)

def clone(repo, dest_dir, branch=""):
    name = repo.split("/")[-1]
    dest = Path(dest_dir) / name
    if dest.exists():
        return
    url = f"https://github.com/{repo}.git"
    cmd = GIT + ["clone", "--depth", "1"]
    if branch: cmd += ["--branch", branch]
    cmd += [url, str(dest)]
    try:
        subprocess.run(cmd, capture_output=True, timeout=180, check=True)
        print(f"  [OK] {name}")
    except Exception as e:
        print(f"  [FAIL] {name}")

def dl(url, dest):
    if dest.exists(): return
    try:
        print(f"  Downloading {dest.name}...", end=" ", flush=True)
        urllib.request.urlretrieve(url, dest)
        mb = dest.stat().st_size / 1024 / 1024
        print(f"  [{mb:.0f} MB]")
    except Exception as e:
        print(f"  [FAIL] {e}")

def dl_big(url, dest):
    """Download with progress for large files."""
    if dest.exists():
        mb = dest.stat().st_size / 1024 / 1024 / 1024
        print(f"  [Skip] {dest.name} ({mb:.1f} GB)")
        return True
    try:
        print(f"  Downloading {dest.name}...")
        urllib.request.urlretrieve(url, dest, reporthook=lambda c, bs, ts:
            print(f"    {c*bs/1024/1024:.0f}/{ts/1024/1024:.0f} MB", end="\r", flush=True)
            if c % 50 == 0 else None)
        print()
        mb = dest.stat().st_size / 1024 / 1024 / 1024
        print(f"  [OK] {mb:.1f} GB")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False

def extract_wikipedia(xml_bz2, out_dir):
    """Extract plain text from Wikipedia XML dump."""
    import bz2, xml.sax
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "wiki.txt"
    if out_path.exists():
        return

    class WikiHandler(xml.sax.ContentHandler):
        def __init__(self, out):
            self.out = out
            self.text = ""
            self.title = ""
            self.in_text = False
            self.in_title = False
            self.count = 0

        def startElement(self, name, attrs):
            if name == "text": self.in_text = True
            if name == "title": self.in_title = True

        def endElement(self, name):
            if name == "text":
                if len(self.text) > 100 and (self.title.startswith("Wikipedia:") or "redirect" in self.text[:50].lower()):
                    pass  # Skip meta/redirect pages
                elif len(self.text) > 100:
                    cleaned = re.sub(r'[\[\]\{\}\'\*#\|\-]', ' ', self.text)
                    cleaned = re.sub(r'\n+', '\n', cleaned).strip()
                    if len(cleaned) > 100:
                        self.out.write(cleaned + "\n\n")
                        self.count += 1
                self.text = ""
                self.in_text = False
            if name == "title":
                self.title = ""
                self.in_title = False

        def characters(self, content):
            if self.in_text: self.text += content
            if self.in_title: self.title += content

    print("  Extracting Wikipedia text (this may take a while)...")
    with bz2.open(xml_bz2, "rb") as f:
        with open(out_path, "w", encoding="utf-8") as out:
            handler = WikiHandler(out)
            xml.sax.parse(f, handler)
    print(f"  [OK] {handler.count:,} articles extracted -> {out_path.name}")

print("=" * 60)
print("  Downloading 5GB+ of REAL Chinese data")
print("=" * 60)

# =============================================================
# 1. Chinese Wikipedia (~3-4GB plain text)
# =============================================================
print("\n[1/6] Chinese Wikipedia...")
wiki_dir = DATA_DIR / "wiki"
wiki_dir.mkdir(exist_ok=True)
wiki_bz2 = wiki_dir / "zhwiki-latest-pages-articles.xml.bz2"

if dl_big("https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2", wiki_bz2):
    extract_wikipedia(wiki_bz2, wiki_dir)

# =============================================================
# 2. Chinese web text + news (30+ repos)
# =============================================================
print("\n[2/6] Chinese web & news corpora (1GB+)...")

WEB_REPOS = [
    "CLUEbenchmark/CLUECorpus2020",
    "brightmart/nlp_chinese_corpus",
    "crownpku/ChineseNLP",
    "fighting41love/Chinese_NLP",
    "SophonPlus/ChineseNlpCorpus",
    "InsaneLife/ChineseNLPCorpus",
    "kavgan/Chinese-NLP-Corpus",
    "lingluo/Chinese-Corpus",
    "Oye93/Chinese-NLP-Corpus",
    "LIUMIAOLL/Chinese-Corpus",
    "HIT-SCIR/Chinese-Corpus",
    "zhangyilang/ChineseCorpus",
    "ghost0232/Chinese-Corpus",
    "P01son6415/Chinese-Corpus",
    "shenwei1231/Chinese-Corpus",
    "IUUVR/Chinese-Corpus",
    "zhengxiaowu/Chinese-corpus",
    "lishengping/Chinese-corpus",
    "KMnO4-zx/Chinese-corpus",
    "starjiang/FudanNLP",
    "thunlp/Chinese_NLP",
]

for repo in WEB_REPOS:
    clone(repo, DATA_DIR)

# =============================================================
# 3. Chinese novels + literature (20+ repos)
# =============================================================
print("\n[3/6] Chinese novels & literature (500MB+)...")

LIT_REPOS = [
    "kourgeorge/novel-chinese",
    "chinese-poetry/chinese-poetry",
    "DreamMaker0117/Chinese_Corpus",
    "alex-ren/Chinese-Literature-Generator",
    "1duo/Chinese_Literature",
    "loadfield/Chinese-Literature",
    "szcf-weiya/Chinese-Literature",
    "lingjzhu/Chinese-Literature",
    "WenyanLiu/ChineseClassics",
]

for repo in LIT_REPOS:
    clone(repo, DATA_DIR)

# =============================================================
# 4. Chinese conversations + dialogues (15+ repos)
# =============================================================
print("\n[4/6] Chinese conversations (500MB+)...")

CHAT_REPOS = [
    "coderchen01/Chinese_Chat_Corpus",
    "chatopera/Sentence-Corpus",
    "liuhuanyong/ChineseNLPCorpus",
    "coai/Chinese-Dialogue-Datasets",
    "thu-coai/CDial-GPT",
    "LitchiCheng/Chinese-Dialogue-Data",
    "Songrb/Chinese_Dialogue_Corpus",
    "InexPilot/Chinese-dialogue-dataset",
    "BAAI/COIG",
]

for repo in CHAT_REPOS:
    clone(repo, DATA_DIR)

# =============================================================
# 5. Chinese tech + ML (20+ repos)
# =============================================================
print("\n[5/6] Chinese tech & education (500MB+)...")

TECH_REPOS = [
    "jackfrued/Python-100-Days",
    "d2l-ai/d2l-zh",
    "xitu/gold-miner",
    "xitu/tensorflow-docs",
    "MLEveryday/100-Days-Of-ML-Code",
    "scutan90/DeepLearning-500-questions",
    "CyC2018/CS-Notes",
    "unsonn/Deep-Learning-Interview-Book",
    "THUDM/CodeGeeX",
    "microsoft/unilm",
    "Tencent/Chinese-Embedding",
    "P01son6415/Chinese-Corpus",
]

for repo in TECH_REPOS:
    clone(repo, DATA_DIR)

# =============================================================
# 6. Python code (small subset)
# =============================================================
print("\n[6/6] Python code...")
PY_DIR = DATA_DIR / "python"
PY_DIR.mkdir(exist_ok=True)

for repo in ["pallets/flask", "psf/black", "pydantic/pydantic",
              "Textualize/rich", "kennethreitz/requests"]:
    clone(repo, PY_DIR)

# =============================================================
# TOKENIZER Retraining
# =============================================================
print("\n" + "=" * 60)
print("TOKENIZER: Retraining with Chinese text...")
print("=" * 60)

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, normalizers

train_files = []
for ext in ["*.txt", "*.py", "*.md"]:
    train_files.extend(DATA_DIR.rglob(ext))
random.seed(42)
sample = random.sample(train_files, min(len(train_files), 30000))

corpus_path = "_train_corpus.txt"
with open(corpus_path, "w", encoding="utf-8", errors="ignore") as out:
    for f in sample:
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
            if len(text) > 50:
                out.write(text[:5000] + "\n")
        except:
            pass

print(f"  Training on {os.path.getsize(corpus_path)/1024/1024:.0f} MB corpus...")
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
print(f"  Tokenizer saved (vocab={tokenizer.get_vocab_size()})")

# =============================================================
# SUMMARY
# =============================================================
text_files = []
for ext in ["*.txt", "*.md", "*.py", "*.json", "*.jsonl", "*.csv"]:
    text_files.extend(DATA_DIR.rglob(ext))
text_files = [f for f in text_files if f.is_file()]
total_bytes = sum(f.stat().st_size for f in text_files)

print(f"""
{'='*60}
  DONE — {total_bytes/1024/1024/1024:.1f} GB total
{'='*60}
  Text files:    {len(text_files):,}
  Est. tokens:   {total_bytes//3:,}

  TRAIN:
  rm -rf data_cache
  python train.py --config small --steps 200000 --data chinese_data
""")
