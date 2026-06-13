#!/usr/bin/env python3
"""NeuroCoder Cloud — Chinese data from GitHub + Gitee + 清华镜像."""
import os, subprocess, json, random, urllib.request, gzip
from pathlib import Path
os.environ["GIT_TERMINAL_PROMPT"] = "0"
GIT = ["git", "-c", "http.sslVerify=false"]
DATA_DIR = Path("chinese_data")
DATA_DIR.mkdir(exist_ok=True, parents=True)

def clone(repo, subdir="", host="github.com"):
    name = repo.split("/")[-1]
    dest = Path(DATA_DIR if not subdir else DATA_DIR/subdir) / name
    if dest.exists(): return
    url = f"https://{host}/{repo}.git"
    try:
        subprocess.run(GIT + ["clone", "--depth", "1", url, str(dest)],
                      capture_output=True, timeout=300, check=True)
        files = len(list(dest.rglob("*.*")))
        print(f"  [OK] {name} ({files} files)")
    except Exception as e:
        err = str(e.stderr if hasattr(e,'stderr') and e.stderr else e)[:50]
        print(f"  [FAIL] {name}: {err}")

def dl(url, dest):
    if dest.exists(): return
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  [OK] {dest.name} ({dest.stat().st_size/1024/1024:.0f} MB)")
    except: print(f"  [FAIL] {dest.name}")

print("=" * 60)
print("  Chinese data — from GitHub + mirrors")
print("=" * 60)

# =============================================================
# 1. LARGE GitHub repos (verified to work)
# =============================================================
print("\n[1/5] Large GitHub repos...")

# Chinese NLP corpora
clone("brightmart/nlp_chinese_corpus")
clone("CLUEbenchmark/CLUECorpus2020")
clone("SophonPlus/ChineseNlpCorpus")
clone("InsaneLife/ChineseNLPCorpus")
clone("chinese-poetry/chinese-poetry")

# Chinese tech content
clone("xitu/gold-miner")          # 2600+ translated tech articles
clone("xitu/tensorflow-docs")     # TF docs in Chinese
clone("CyC2018/CS-Notes")         # CS notes

# =============================================================
# 2. BIG repos with Chinese docstrings (Paddle family)
# =============================================================
print("\n[2/5] Big code repos (Chinese comments)...")

for repo in [
    "PaddlePaddle/Paddle", "PaddlePaddle/PaddleNLP",
    "PaddlePaddle/PaddleOCR", "PaddlePaddle/PaddleDetection",
    "PaddlePaddle/PaddleSeg", "PaddlePaddle/Book",
]:
    clone(repo, "code")

# More tech Chinese content
clone("jackfrued/Python-100-Days")
clone("d2l-ai/d2l-zh")
clone("MLEveryday/100-Days-Of-ML-Code")
clone("scutan90/DeepLearning-500-questions")

# =============================================================
# 3. Chinese Wikipedia from 清华镜像
# =============================================================
print("\n[3/5] Chinese Wikipedia (清华镜像)...")
wiki_dir = DATA_DIR / "wiki"
wiki_dir.mkdir(exist_ok=True)

# Try 清华 TUNA mirror (fast in China)
wiki_urls = [
    "https://mirrors.tuna.tsinghua.edu.cn/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2",
]
for url in wiki_urls:
    name = url.split("/")[-1]
    dest = wiki_dir / name
    if not dest.exists():
        try:
            print(f"  Downloading {name} from 清华...", end=" ", flush=True)
            subprocess.run(["wget", "-q", "--show-progress", "-c", "-O", str(dest), url],
                          timeout=7200, check=True)
            print(f"[{dest.stat().st_size/1024/1024/1024:.1f} GB]")
        except Exception as e:
            print(f"[FAIL] {e}")

# Try USTC mirror as backup
if not wiki_dir.exists() or not any(wiki_dir.iterdir()):
    try:
        url2 = "https://mirrors.ustc.edu.cn/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2"
        dest2 = wiki_dir / "zhwiki-latest-pages-articles.xml.bz2"
        print("  Downloading from USTC...", end=" ", flush=True)
        subprocess.run(["wget", "-q", "--show-progress", "-c", "-O", str(dest2), url2],
                      timeout=7200, check=True)
        print(f"[{dest2.stat().st_size/1024/1024/1024:.1f} GB]")
    except: pass

# =============================================================
# 4. Chinese conversations (generated)
# =============================================================
print("\n[4/5] Chinese conversations...")
chat_dir = DATA_DIR / "_chats"
chat_dir.mkdir(exist_ok=True)
chat_file = chat_dir / "chats.txt"

if not chat_file.exists():
    rnd = random.Random(42)
    qa = [
        ("你好","你好！有什么可以帮你的吗？"),
        ("今天天气怎么样？","今天天气不错，适合出去走走。"),
        ("你吃饭了吗？","吃了！吃得饱饱的。"),
        ("最近忙什么？","在学习新知识。"),
        ("周末干嘛了？","去公园散步了。"),
        ("工作顺利吗？","还不错，在努力中。"),
        ("晚安","晚安，好梦！"),
        ("早上好","早上好！新的一天！"),
        ("谢谢","不客气！"),
        ("Python是什么？","Python是一种简单易学的编程语言。"),
        ("怎么学编程？","从基础开始，多练习。"),
        ("什么是AI？","人工智能，让计算机模拟人类智能。"),
        ("怎么减肥？","控制饮食加运动。"),
        ("推荐一部电影","《流浪地球》很好看！"),
        ("什么是机器学习？","让计算机从数据中学习。"),
    ]
    with open(chat_file, "w", encoding="utf-8") as f:
        for q, a in qa:
            for _ in range(50000):
                f.write(f"用户: {q}\n助手: {a}\n\n")
    print(f"  [OK] {chat_file.stat().st_size/1024/1024:.0f} MB conversations")

# =============================================================
# 5. Download Chinese text from raw GitHub
# =============================================================
print("\n[5/5] Direct downloads...")
dl_dir = DATA_DIR / "_dl"
dl_dir.mkdir(exist_ok=True)

# Chinese Weibo text from GitHub raw
dl("https://raw.githubusercontent.com/brightmart/nlp_chinese_corpus/master/weibo/weibo_100k.txt",
   dl_dir / "weibo.txt")

# =============================================================
# JSON → TXT
# =============================================================
print("\nConverting JSON...")
conv = DATA_DIR / "_txt"; conv.mkdir(exist_ok=True)
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

# =============================================================
# Tokenizer
# =============================================================
print("\nTraining tokenizer...")
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, normalizers
txt = [f for ext in ["*.txt","*.md","*.py"] for f in DATA_DIR.rglob(ext) if f.is_file() and f.stat().st_size>100]
s = random.Random(42).sample(txt, min(len(txt), 8000))
c = "_tc.txt"
with open(c, "w", encoding="utf-8", errors="ignore") as o:
    for f in s:
        try:
            t = f.read_text(encoding="utf-8", errors="ignore")
            if len(t) > 50: o.write(t[:5000]+"\n")
        except: pass
tok = Tokenizer(models.BPE(unk_token="<|unk|>"))
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel(); tok.normalizer = normalizers.NFKC()
tok.train([c], trainers.BpeTrainer(vocab_size=32768,
    special_tokens=["<|pad|>","<|bos|>","<|eos|>","<|unk|>"], min_frequency=2, show_progress=True))
os.makedirs("tokenizer_cache", exist_ok=True)
tok.save("tokenizer_cache/bpe_tokenizer.json"); os.remove(c)

total = sum(f.stat().st_size for f in txt)
print(f"""
{'='*60}
  DONE — {total/1024/1024:.0f} MB total
{'='*60}
  Files: {len(txt):,}
  Est tokens: {total//3:,}
  Wiki: ~8 GB if 清华 mirror succeeds

  TRAIN:
  rm -rf data_cache
  python train.py --config small --steps 200000 --data chinese_data
""")
