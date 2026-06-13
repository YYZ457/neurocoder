#!/usr/bin/env python3
"""NeuroCoder Cloud — Chinese data from ModelScope (LFS works now!)."""
import os, subprocess, json, random, shutil
from pathlib import Path
os.environ["GIT_TERMINAL_PROMPT"] = "0"
DATA_DIR = Path("chinese_data"); DATA_DIR.mkdir(exist_ok=True, parents=True)
os.makedirs("tokenizer_cache", exist_ok=True)

def clone_ms(dataset, subdir=""):
    name = dataset.split("/")[-1]
    dest = Path(DATA_DIR if not subdir else DATA_DIR/subdir) / name
    if dest.exists(): print(f"  [Skip] {name}"); return str(dest)
    url = f"https://www.modelscope.cn/datasets/{dataset}.git"
    try:
        subprocess.run(["git", "clone", "--depth", "1", url, str(dest)],
                      capture_output=True, timeout=600, check=True)
        subprocess.run(["git", "-C", str(dest), "lfs", "pull"], capture_output=True, timeout=600)
        files = [f for f in dest.rglob("*") if f.is_file()]
        mb = sum(f.stat().st_size for f in files) / 1024 / 1024
        print(f"  [OK] {name} ({len(files)} files, {mb:.0f} MB)")
        return str(dest)
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        return None

def extract_convs(src, out_dir, name):
    """Extract conversations from SFT JSONL data."""
    out_dir = Path(out_dir); out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{name}.txt"
    if out_path.exists(): return 0
    src = Path(src)
    if not src.exists(): return 0

    files = list(src.rglob("*.jsonl")) + list(src.rglob("*.json"))
    if not files:
        files = list(src.rglob("*"))
    files = [f for f in files if f.is_file() and f.stat().st_size > 100]

    count = 0
    with open(out_path, "w", encoding="utf-8") as fo:
        for f in files:
            try:
                for line in open(f, encoding="utf-8", errors="ignore"):
                    line = line.strip()
                    if not line: continue
                    try:
                        item = json.loads(line)
                    except:
                        if len(line) > 20: fo.write(line + "\n"); count += 1
                        continue
                    if not isinstance(item, dict): continue
                    # Try all known SFT field combos
                    inst = item.get("instruction") or item.get("q") or item.get("question") or item.get("query") or ""
                    out_ = item.get("output") or item.get("a") or item.get("answer") or item.get("response") or ""
                    if inst and out_:
                        fo.write(f"用户: {inst}\n助手: {out_}\n\n"); count += 1
                    elif item.get("conversation"):
                        for t in item["conversation"]:
                            h = t.get("human") or t.get("user") or ""
                            b = t.get("assistant") or t.get("bot") or ""
                            if h and b: fo.write(f"用户: {h}\n助手: {b}\n\n"); count += 1
                    elif item.get("messages"):
                        msgs = item["messages"]
                        for i in range(0, len(msgs)-1, 2):
                            if i+1 < len(msgs):
                                fo.write(f"用户: {msgs[i].get('content','')}\n助手: {msgs[i+1].get('content','')}\n\n"); count += 1
                    else:
                        texts = [str(v) for v in item.values() if isinstance(v, str) and len(v) > 20]
                        if texts: fo.write("\n".join(texts) + "\n\n"); count += 1
            except: pass
    if count == 0: out_path.unlink(missing_ok=True)
    return count

print("=" * 60)
print("  Chinese data from ModelScope (LFS enabled)")
print("=" * 60)

# =============================================================
# 1. qiaojiedongfeng — user's dataset
# =============================================================
print("\n[1/4] qiaojiedongfeng...")
path = clone_ms("qiaojiedongfeng/qiaojiedongfeng", "qjd")
if path:
    c = extract_convs(path, DATA_DIR / "_txt", "qjd")
    print(f"    -> {c} conversations" if c else "    (no convs extracted)")

# =============================================================
# 2. deepctrl-sft-data — 11.38M Chinese SFT conversations
# =============================================================
print("\n[2/4] deepctrl-sft-data (11.38M Chinese conversations)...")
path = clone_ms("AI-ModelScope/deepctrl-sft-data", "sft")
if path:
    c = extract_convs(path, DATA_DIR / "_txt", "deepctrl_sft")
    print(f"    -> {c:,} conversations" if c else "    (no convs extracted)")

# =============================================================
# 3. SFT-Chinese-Dataset — Firefly 1.1M + ShareGPT
# =============================================================
print("\n[3/4] SFT-Chinese-Dataset...")
path = clone_ms("zhuangxialie/SFT-Chinese-Dataset", "sft2")
if path:
    c = extract_convs(path, DATA_DIR / "_txt", "sft_chinese")
    print(f"    -> {c:,} conversations" if c else "    (no convs extracted)")

# =============================================================
# 4. Backup: GitHub verified repos + conversations
# =============================================================
print("\n[4/4] Generating conversation backup...")
import urllib.request
GIT = ["git", "-c", "http.sslVerify=false"]

for repo in ["xitu/gold-miner", "chinese-poetry/chinese-poetry",
              "jackfrued/Python-100-Days", "SophonPlus/ChineseNlpCorpus"]:
    name = repo.split("/")[-1]
    dest = DATA_DIR / "backup" / name
    if not dest.exists():
        try:
            subprocess.run(GIT + ["clone", "--depth", "1", f"https://github.com/{repo}.git", str(dest)],
                          capture_output=True, timeout=300)
        except: pass

# Generate conversations
chat_f = DATA_DIR / "_chats" / "chats.txt"
if not chat_f.exists():
    Path(DATA_DIR / "_chats").mkdir(exist_ok=True)
    rnd = random.Random(42)
    qa = [("你好","你好！有什么可以帮助你的吗？"),("早上好","早上好！"),
           ("今天天气怎么样？","今天天气不错！"),("Python是什么？","Python是一种编程语言。")]
    with open(chat_f, "w", encoding="utf-8") as f:
        for q, a in qa:
            for _ in range(50000): f.write(f"用户: {q}\n助手: {a}\n\n")

# =============================================================
# Tokenizer
# =============================================================
print("\nTraining tokenizer...")
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, normalizers
txt = [f for ext in ["*.txt","*.md"] for f in DATA_DIR.rglob(ext) if f.is_file() and f.stat().st_size>100]
random.shuffle(txt)
c = "_tc.txt"
with open(c, "w", encoding="utf-8", errors="ignore") as o:
    for f in txt[:10000]:
        try:
            t = f.read_text(encoding="utf-8", errors="ignore")
            if len(t) > 50: o.write(t[:5000]+"\n")
        except: pass
tok = Tokenizer(models.BPE(unk_token="<|unk|>"))
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel(); tok.normalizer = normalizers.NFKC()
tok.train([c], trainers.BpeTrainer(vocab_size=32768,
    special_tokens=["<|pad|>","<|bos|>","<|eos|>","<|unk|>"], min_frequency=2, show_progress=True))
tok.save("tokenizer_cache/bpe_tokenizer.json"); os.remove(c)

total = sum(f.stat().st_size for f in txt)
print(f"""
{'='*60}
  DONE — {total/1024/1024:.0f} MB
{'='*60}
  TRAIN:
  rm -rf data_cache
  python train.py --config small --steps 200000 --data chinese_data
""")
