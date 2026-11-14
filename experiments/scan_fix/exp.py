import sys, time, json, glob, os, random, importlib.util, math
import torch
variant, steps, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
here = os.path.dirname(os.path.abspath(__file__))
root = {'old': os.path.join(here, 'old'), 'new': os.path.abspath(os.path.join(here, '..', '..'))}[variant]
sys.path.insert(0, root)
from config import NeuroCoderConfig
from model import NeuroCoder
torch.set_num_threads(2)

# --- data: Python stdlib source as bytes ---
files = sorted(glob.glob('/usr/lib/python3*/**/*.py', recursive=True))
random.Random(0).shuffle(files)
def load(fs, cap):
    buf = bytearray()
    for f in fs:
        try: buf += open(f, 'rb').read() + b'\n'
        except Exception: pass
        if len(buf) >= cap: break
    return torch.tensor(list(buf[:cap]), dtype=torch.long)
train = load(files[:-200], 8_000_000); val = load(files[-200:], 500_000)
# shift bytes by +3 so pad/bos/eos ids 0..2 never collide with data
train += 3; val += 3

cfg = NeuroCoderConfig(vocab_size=259, d_model=96, n_blocks=4, n_experts=8, n_active_experts=2,
    expert_dim=256, ssm_state_dim=16, ssm_dt_rank=32, max_seq_len=128, dropout=0.0,
    expert_dropout=0.0, focal_loss_gamma=0.0, use_gradient_checkpointing=False, use_bf16=False)
torch.manual_seed(0)
model = NeuroCoder(cfg)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)
B, L = 8, 128
g = torch.Generator().manual_seed(1)
def batch(data, gen):
    ix = torch.randint(0, len(data) - L, (B,), generator=gen)
    return torch.stack([data[i:i+L] for i in ix])
@torch.no_grad()
def evaluate():
    model.eval(); gv = torch.Generator().manual_seed(123); tot = 0; pos = torch.zeros(L-1)
    for _ in range(10):
        x = batch(val, gv); logits = model(x)['logits'][:, :-1]
        ce = torch.nn.functional.cross_entropy(logits.reshape(-1, 259), x[:, 1:].reshape(-1), reduction='none').view(B, L-1)
        tot += ce.mean().item(); pos += ce.mean(0)
    model.train(); return tot / 10, pos / 10
log = []; t0 = time.time()
for step in range(1, steps + 1):
    lr = 1e-3 * min(1, step / 100) * (0.5 * (1 + math.cos(math.pi * step / steps)) * 0.9 + 0.1)
    for pg in opt.param_groups: pg['lr'] = lr
    x = batch(train, g); o = model(x, labels=x)
    opt.zero_grad(); o['loss'].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    if step % 50 == 0 or step == 1:
        v, pos = evaluate()
        rec = dict(step=step, train=o['ce_raw'].item(), val=v, sec=time.time()-t0,
                   val_pos=[pos[:8].mean().item(), pos[8:32].mean().item(), pos[32:64].mean().item(), pos[64:].mean().item()])
        log.append(rec); print(variant, json.dumps(rec), flush=True)
json.dump(log, open(out, 'w'))
