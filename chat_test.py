"""Chat with the trained model. Adds conversation prefix automatically."""
import sys; sys.path.insert(0,'.')
import torch
from data import CodeTokenizer
from model import NeuroCoder
from pathlib import Path

# Load latest checkpoint
ckpt_dir = Path("checkpoints")
ckpt_files = sorted(ckpt_dir.glob("*.pt"), key=lambda f: f.stat().st_mtime, reverse=True)
if not ckpt_files:
    print("No checkpoint found!")
    sys.exit(1)

ckpt_path = ckpt_files[0]
print(f"Loading {ckpt_path}...")
ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

t = CodeTokenizer()
model = NeuroCoder(ckpt["config"])
model.load_state_dict(ckpt["model_state_dict"])
model = model.cuda().eval()
print("Ready! (type 'quit' to exit)\n")

while True:
    inp = input(">>> ")
    if not inp or inp.lower() in ("quit", "exit", "q"):
        break

    # CRITICAL: wrap input in conversation format so model knows the context
    prompt = f"用户: {inp}\n助手: "

    ids = t.encode(prompt, max_length=128)
    inp_tensor = torch.tensor([ids], device="cuda")

    with torch.no_grad():
        out = model.generate(inp_tensor, max_new_tokens=100, temperature=0.5, top_p=0.9)

    full_text = t.decode(out[0].tolist())
    response = full_text[len(prompt):]

    # Trim at next user turn or EOS
    for stop in ["用户:", "用户：", "<|eos|>"]:
        if stop in response:
            response = response[:response.index(stop)]

    print(response.strip())
    print()
