#!/usr/bin/env python3
"""NeuroCoder Cloud — Chinese data from ModelScope (国内高速下载)."""
import os, subprocess, json, random, shutil
from pathlib import Path
os.environ["GIT_TERMINAL_PROMPT"] = "0"
DATA_DIR = Path("chinese_data")
DATA_DIR.mkdir(exist_ok=True, parents=True)

print("=" * 60)
print("  Installing git-lfs and modelscope...")
print("=" * 60)
subprocess.run(["apt-get", "install", "-y", "git-lfs"], capture_output=True)
subprocess.run(["git", "lfs", "install"], capture_output=True)
subprocess.run(["pip", "install", "-q", "modelscope"], check=False)

def clone_ms(dataset, subdir=""):
    """Clone a dataset from ModelScope."""
    name = dataset.split("/")[-1]
    dest = Path(DATA_DIR if not subdir else DATA_DIR/subdir) / name
    if dest.exists():
        print(f"  [Skip] {name}")
        return True
    url = f"https://www.modelscope.cn/datasets/{dataset}.git"
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", url, str(dest)],
            capture_output=True, timeout=600, check=True
        )
        # Pull LFS files (try multiple methods)
        subprocess.run(["git", "-C", str(dest), "lfs", "pull"], capture_output=True, timeout=600)
        # Try fetching LFS if pull didn't work
        subprocess.run(["git", "-C", str(dest), "lfs", "fetch", "--all"], capture_output=True, timeout=300)
        subprocess.run(["git", "-C", str(dest), "checkout", "."], capture_output=True, timeout=300)
        files = [f for f in dest.rglob("*") if f.is_file()]
        total_mb = sum(f.stat().st_size for f in files) / 1024 / 1024
        print(f"  [OK] {name} ({len(files)} files, {total_mb:.0f} MB)")
    except Exception as e:
        err = str(e.stderr if hasattr(e,'stderr') and e.stderr else e)[:80]
        print(f"  [FAIL] {name}: {err}")
        return False

def jsonl_to_txt(src_dir, out_dir, name):
    """Convert SFT/conversation data to plain text. Handles JSONL, JSON, TXT."""
    out_dir = Path(out_dir); out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{name}.txt"
    if out_path.exists(): return

    src = Path(src_dir)
    if not src.exists() or sum(f.stat().st_size for f in src.rglob("*") if f.is_file()) < 1000:
        return

    # Collect all files
    files = list(src.rglob("*.jsonl")) + list(src.rglob("*.json")) + list(src.rglob("*.txt"))
    if not files:
        files = [f for f in src.rglob("*") if f.is_file() and f.stat().st_size > 100]

    count = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for f in files:
            try:
                for line in open(f, encoding="utf-8", errors="ignore"):
                    line = line.strip()
                    if not line: continue
                    try:
                        item = json.loads(line)
                    except:
                        # Plain text line
                        if len(line) > 20:
                            out.write(line + "\n")
                            count += 1
                        continue

                    if not isinstance(item, dict): continue

                    # Try all common Chinese SFT field combinations
                    inst = (item.get("instruction") or item.get("q") or item.get("question") or
                            item.get("query") or item.get("input") or "")
                    output = (item.get("output") or item.get("a") or item.get("answer") or
                              item.get("response") or item.get("target") or "")

                    if inst and output:
                        out.write(f"用户: {inst}\n助手: {output}\n\n")
                        count += 1
                    elif item.get("conversation") or item.get("history"):
                        conv = item.get("conversation") or item.get("history") or []
                        for turn in conv:
                            h = turn.get("human") or turn.get("user") or turn.get("q") or ""
                            b = turn.get("assistant") or turn.get("bot") or turn.get("a") or ""
                            if h and b:
                                out.write(f"用户: {h}\n助手: {b}\n\n")
                                count += 1
                    elif item.get("messages"):
                        msgs = item.get("messages", [])
                        for i in range(0, len(msgs)-1, 2):
                            if i+1 < len(msgs):
                                out.write(f"用户: {msgs[i].get('content','')}\n助手: {msgs[i+1].get('content','')}\n\n")
                                count += 1
                    else:
                        texts = [str(v) for v in item.values() if isinstance(v, str) and len(v) > 20]
                        if texts:
                            out.write("\n".join(texts) + "\n\n")
                            count += 1
            except: pass

    if count > 0:
        mb = out_path.stat().st_size / 1024 / 1024
        print(f"    -> {name}.txt: {count:,} ({mb:.0f} MB)")
    else:
        out_path.unlink(missing_ok=True)

print("=" * 60)
print("  Downloading Chinese data from ModelScope")
print("=" * 60)

# =============================================================
# 1. qiaojiedongfeng — 用户指定的中文对话数据集
# =============================================================
print("\n[1/3] qiaojiedongfeng (中文对话数据集)...")
qjd_dir = DATA_DIR / "qjd"
qjd_dir.mkdir(exist_ok=True)

if clone_ms("qiaojiedongfeng/qiaojiedongfeng", "qjd"):
    # Auto-convert whatever format
    jsonl_to_txt(DATA_DIR / "qjd" / "qiaojiedongfeng", DATA_DIR / "_txt", "qjd")

# =============================================================
# 2. deepctrl-sft-data — 11.38M Chinese SFT conversations
# =============================================================
print("\n[2/3] deepctrl-sft-data (1138万条中文对话)...")
sft_dir = DATA_DIR / "sft"
sft_dir.mkdir(exist_ok=True)

if clone_ms("AI-ModelScope/deepctrl-sft-data", "sft"):
    jsonl_to_txt(DATA_DIR / "sft" / "deepctrl-sft-data", DATA_DIR / "_txt", "deepctrl_sft")

# =============================================================
# 3. Generate conversation backup
# =============================================================
print("\n[3/3] Generating conversation backup...")
chat_file = DATA_DIR / "_chats" / "chats.txt"
if not chat_file.exists():
    Path(DATA_DIR / "_chats").mkdir(exist_ok=True)
    rnd = random.Random(42)
    qa = [
        ("你好","你好！有什么可以帮你的吗？"),
        ("今天天气怎么样？","今天天气不错！"),
        ("你吃饭了吗？","吃了！"),
        ("晚安","晚安，好梦！"),
        ("Python是什么？","Python是一种编程语言。"),
        ("什么是AI？","人工智能。"),
    ]
    with open(chat_file, "w", encoding="utf-8") as f:
        for q, a in qa:
            for _ in range(30000):
                f.write(f"用户: {q}\n助手: {a}\n\n")
    print(f"  [OK] conversations ({chat_file.stat().st_size/1024/1024:.0f} MB)")

# =============================================================
# Tokenizer
# =============================================================
print("\nTraining tokenizer...")
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, normalizers
txt = [f for ext in ["*.txt","*.md"] for f in DATA_DIR.rglob(ext) if f.is_file() and f.stat().st_size>100]
s = random.Random(42).sample(txt, min(len(txt), 8000))
c = "_tc.txt"
with open(c, "w", encoding="utf-8", errors="ignore") as o:
    for f in s:
        try:
            t = f.read_text(encoding="utf-8", errors="ignore")
            if len(t) > 50: o.write(t[:5000] + "\n")
        except: pass
tok = Tokenizer(models.BPE(unk_token="<|unk|>"))
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()
tok.normalizer = normalizers.NFKC()
tok.train([c], trainers.BpeTrainer(vocab_size=32768,
    special_tokens=["<|pad|>","<|bos|>","<|eos|>","<|unk|>"], min_frequency=2, show_progress=True))
os.makedirs("tokenizer_cache", exist_ok=True)
tok.save("tokenizer_cache/bpe_tokenizer.json")
os.remove(c)

total = sum(f.stat().st_size for f in txt)
print(f"""
{'='*60}
  DONE — {total/1024/1024:.0f} MB
{'='*60}
  Files: {len(txt):,}

  TRAIN:
  rm -rf data_cache
  python train.py --config small --steps 200000 --data chinese_data
""")
