#!/usr/bin/env python3
"""NeuroCoder Cloud — Chinese conversation model data."""
import os, subprocess, urllib.request, json, re, random
from pathlib import Path

os.environ["GIT_TERMINAL_PROMPT"] = "0"
GIT = ["git", "-c", "http.sslVerify=false"]
DATA_DIR = Path("chinese_data")
DATA_DIR.mkdir(exist_ok=True, parents=True)

def clone(repo, subdir=""):
    name = repo.split("/")[-1]
    dest = Path(DATA_DIR if not subdir else DATA_DIR/subdir) / name
    if dest.exists(): return
    try:
        subprocess.run(GIT + ["clone", "--depth", "1", f"https://github.com/{repo}.git", str(dest)],
                      capture_output=True, timeout=300, check=True)
        print(f"  [OK] {name}")
    except: print(f"  [FAIL] {name}")

def dl(url, dest):
    if dest.exists() and dest.stat().st_size > 1024: return
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  [OK] {dest.name} ({dest.stat().st_size/1024/1024:.0f} MB)")
    except: print(f"  [FAIL] {dest.name}")

# Try to install git-lfs for large files
subprocess.run(["apt-get", "install", "-y", "git-lfs"], capture_output=True)
subprocess.run(["git", "lfs", "install"], capture_output=True)

def clone_lfs(repo, subdir=""):
    name = repo.split("/")[-1]
    dest = Path(DATA_DIR if not subdir else DATA_DIR/subdir) / name
    if dest.exists(): return
    try:
        subprocess.run(GIT + ["clone", "--depth", "1", f"https://github.com/{repo}.git", str(dest)],
                      capture_output=True, timeout=300, check=True)
        # Pull LFS files
        subprocess.run(["git", "-C", str(dest), "lfs", "pull"], capture_output=True, timeout=600)
        size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
        print(f"  [OK] {name} ({size/1024/1024:.0f} MB)")
    except Exception as e:
        print(f"  [FAIL] {name}: {str(e)[:60]}")

print("=" * 60)
print("  Chinese Conversation Model — Data Download")
print("=" * 60)

# =============================================================
# 1. Large Chinese NLP datasets (with LFS!)
# =============================================================
print("\n[1/5] Large Chinese NLP datasets...")

# nlp_chinese_corpus — has Weibo, news, Wikipedia (~200MB+ with LFS)
clone_lfs("brightmart/nlp_chinese_corpus")

# CLUECorpus — large news dataset
clone_lfs("CLUEbenchmark/CLUECorpus2020")

# Chinese NLP collections (smaller)
clone("SophonPlus/ChineseNlpCorpus")
clone("InsaneLife/ChineseNLPCorpus")

# =============================================================
# 2. Chinese conversation
# =============================================================
print("\n[2/5] Chinese conversations...")
clone("thu-coai/CDial-GPT")
clone("liuhuanyong/ChineseNLPCorpus")

# =============================================================
# 3. Chinese tech articles (conversational text)
# =============================================================
print("\n[3/5] Chinese tech articles...")
clone("xitu/gold-miner")
clone("xitu/tensorflow-docs")
clone("CyC2018/CS-Notes")

# =============================================================
# 4. Chinese literature + culture
# =============================================================
print("\n[4/5] Chinese literature...")
clone("chinese-poetry/chinese-poetry")

# =============================================================
# 5. Download pre-processed Chinese text directly (no LFS)
# =============================================================
print("\n[5/5] Direct downloads (fast sources)...")

dl_dir = DATA_DIR / "_dl"
dl_dir.mkdir(exist_ok=True)

# Chinese Weibo data (from HuggingFace mirrors)
urls = [
    "https://raw.githubusercontent.com/brightmart/nlp_chinese_corpus/master/weibo/weibo_100k.txt",
]
for url in urls:
    dl(url, dl_dir / url.split("/")[-1])

# Generate conversation data as fallback
print("  Generating Chinese conversation data...")
chat_file = dl_dir / "_generated_chats.txt"
if not chat_file.exists():
    import random as rnd
    rnd.seed(42)
    greetings = ["你好","你好呀","早上好","晚上好","嗨","hello"]
    questions = ["今天天气怎么样？","你吃饭了吗？","在干嘛呢？","心情怎么样？","最近忙什么？"]
    answers = ["挺好的！你呢？","刚吃完，你呢？","在想问题呢","还不错！""挺忙的，不过还好"]
    topics = ["人工智能","Python","电影","音乐","旅游","美食","健身","读书"]

    with open(chat_file, "w", encoding="utf-8") as f:
        for i in range(50000):
            g = rnd.choice(greetings)
            q = rnd.choice(questions)
            a = rnd.choice(answers)
            t = rnd.choice(topics)
            f.write(f"用户: {g}\n助手: 你好！有什么可以帮助你的吗？\n\n")
            f.write(f"用户: {q}\n助手: {a}\n\n")
            f.write(f"用户: 你喜欢{t}吗？\n助手: 喜欢！{t}很有趣。\n\n")
    print(f"  [OK] generated 50K chat turns ({chat_file.stat().st_size/1024:.0f} KB)")

# =============================================================
# JSON to TXT conversion
# =============================================================
print("\nConverting JSON to text...")
conv = DATA_DIR / "_txt"
conv.mkdir(exist_ok=True)
for f in list(DATA_DIR.rglob("*.jsonl")) + list(DATA_DIR.rglob("*.json")):
    if "node_modules" in str(f): continue
    out = conv / f"{f.parent.name}_{f.stem}.txt"
    if out.exists(): continue
    c = 0
    try:
        with open(out, "w", encoding="utf-8") as fo:
            for line in open(f, encoding="utf-8", errors="ignore"):
                try:
                    item = json.loads(line)
                    txt = " ".join(str(v) for v in item.values() if isinstance(v, str) and len(v)>5)
                    if txt: fo.write(txt+"\n"); c+=1
                except: pass
        if not c: out.unlink()
    except: pass

# Collect ALL text files for tokenizer
txt_files = list(DATA_DIR.rglob("*.txt")) + list(DATA_DIR.rglob("*.md"))
txt_files = [f for f in txt_files if f.is_file()]
total_bytes = sum(f.stat().st_size for f in txt_files)

print(f"\nRaw data: {total_bytes/1024/1024:.0f} MB in {len(txt_files)} files")

# =============================================================
# Tokenizer
# =============================================================
print("\nTraining Chinese tokenizer...")
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, normalizers

files = [f for f in txt_files if f.stat().st_size > 100]
sample = random.sample(files, min(len(files), 30000))

corpus = "_tc.txt"
with open(corpus, "w", encoding="utf-8", errors="ignore") as out:
    for f in sample:
        try:
            t = f.read_text(encoding="utf-8", errors="ignore")
            if len(t) > 50: out.write(t[:5000]+"\n")
        except: pass

tok = Tokenizer(models.BPE(unk_token="<|unk|>"))
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()
tok.normalizer = normalizers.NFKC()
trainer = trainers.BpeTrainer(vocab_size=32768,
    special_tokens=["<|pad|>","<|bos|>","<|eos|>","<|unk|>"],
    min_frequency=2, show_progress=True)
tok.train([corpus], trainer)
os.makedirs("tokenizer_cache", exist_ok=True)
tok.save("tokenizer_cache/bpe_tokenizer.json")
os.remove(corpus)

print(f"""
{'='*60}
  DONE — {total_bytes/1024/1024:.0f} MB Chinese text
{'='*60}
  Files: {len(txt_files):,}
  Est. tokens: {total_bytes//3:,}

  TRAIN:
  rm -rf data_cache
  python train.py --config small --steps 200000 --data chinese_data
""")
