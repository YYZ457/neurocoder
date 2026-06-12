import sys; sys.path.insert(0,'.')
import torch
from data import CodeTokenizer
from model import NeuroCoder
from config import CONFIG_4060

t = CodeTokenizer()
checkpoint = torch.load("checkpoints/crash_recovery.pt", map_location="cuda", weights_only=False)

cfg = checkpoint.get("config", CONFIG_4060)
model = NeuroCoder(cfg)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.cuda().eval()

prompt = "def is_palindrome(s):\n    "
input_ids = t.encode(prompt, max_length=512)
inp = torch.tensor([input_ids], device="cuda")

# Generate with low temp for determinism
with torch.no_grad():
    out = model.generate(inp, max_new_tokens=50, temperature=0.3, top_p=0.9)

print(f"Input shape: {inp.shape}")
print(f"Output shape: {out.shape}")
print(f"Full output ids: {out[0][:30].tolist()}")

gen = t.decode(out[0].tolist())
print(f"\nFull text: {repr(gen)}")
print(f"\nGenerated part: {repr(gen[len(prompt):]) if gen.startswith(prompt) else repr(gen)}")
