"""Convert deepctrl JSONL → conversation text for training."""
import json, os
from pathlib import Path

src = "chinese_data/deepctrl/sft_data_zh.jsonl"
out = "chinese_data/deepctrl_converted.txt"

if os.path.exists(out):
    mb = os.path.getsize(out) / 1024 / 1024
    print(f"[Skip] {out} ({mb:.0f} MB)")
else:
    count = 0
    with open(out, "w", encoding="utf-8") as fo:
        with open(src, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                # Multi-turn: history first
                for h in item.get("history", []):
                    if len(h) >= 2 and h[0] and h[1]:
                        fo.write(f"用户: {h[0]}\n助手: {h[1]}\n\n")
                        count += 1
                # Current turn
                user = item.get("input", "") or item.get("instruction", "")
                assistant = item.get("output", "")
                if user and assistant:
                    fo.write(f"用户: {user}\n助手: {assistant}\n\n")
                    count += 1
                if count % 50000 == 0:
                    print(f"  {count:,}...")
    mb = os.path.getsize(out) / 1024 / 1024
    print(f"[OK] {count:,} conversations -> {out} ({mb:.0f} MB)")
