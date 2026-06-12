import sys; sys.path.insert(0,'.')
import torch
from data import CodeTokenizer
from model import NeuroCoder
from config import CONFIG_SMALL

t = CodeTokenizer()
cfg = CONFIG_SMALL

print("Loading latest checkpoint...")
checkpoint = torch.load("D:/NeuroCoder/checkpoints/latest.pt", map_location="cuda", weights_only=False)

model = NeuroCoder(cfg)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.cuda().eval()
print(f"Loaded ({model.get_num_params()/1e6:.1f}M params)\n")

results = []
tests = [
    "def is_palindrome(s):\n    \"\"\"Check if string is palindrome.\"\"\"\n    ",
    "def fibonacci(n):\n    \"\"\"Return nth Fibonacci number.\"\"\"\n    if n <= 0:\n        return 0\n    ",
    "def quick_sort(arr):\n    \"\"\"Sort list using quicksort.\"\"\"\n    if len(arr) <= 1:\n        return arr\n    ",
    "import csv\nimport json\n\ndef read_csv_to_dict(filename):\n    \"\"\"Read CSV and return list of dicts.\"\"\"\n    result = []\n    ",
    "# input a string and check if it is palindrome\ndef is_palindrome(s):\n    ",
]

for prompt in tests:
    input_ids = t.encode(prompt, max_length=512)
    inp = torch.tensor([input_ids], device="cuda")

    with torch.no_grad():
        out = model.generate(inp, max_new_tokens=100, temperature=0.5, top_p=0.9)

    gen = t.decode(out[0].tolist())
    gen_text = gen[len(prompt):] if gen.startswith(prompt) else gen
    results.append(f"{'='*50}\nPrompt: {prompt[:80]}\n{'-'*30}\n{gen_text}\n{'='*50}\n")

# Only write to file, no terminal print
with open("D:/NeuroCoder/test_results.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(results))

print("Written to D:/NeuroCoder/test_results.txt")
