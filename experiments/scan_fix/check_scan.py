import sys, torch, torch.nn.functional as F
sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..', '..'))
from config import NeuroCoderConfig
from model import SelectiveSSM
torch.manual_seed(0)
cfg = NeuroCoderConfig(d_model=64, ssm_state_dim=16, ssm_dt_rank=16)
ssm = SelectiveSSM(cfg)

def ref_scan(u, delta, A, B, C, D):
    Bsz, L, ID = u.shape
    h = torch.zeros(Bsz, ID, A.shape[1])
    ys = []
    for t in range(L):
        dA = torch.exp(delta[:, t, :, None] * A)
        h = dA * h + delta[:, t, :, None] * B[:, t, None, :] * u[:, t, :, None]
        ys.append((h * C[:, t, None, :]).sum(-1) + D * u[:, t])
    return torch.stack(ys, 1)

ID = ssm.inner_dim
for L in [8, 32, 128, 512, 2048]:
    u = torch.randn(1, L, ID)
    delta = F.softplus(torch.randn(1, L, ID) * 0.1 + 0.54)   # init ~ softplus -> ~1
    A = -torch.exp(ssm.A_log.detach())
    B = torch.tanh(torch.randn(1, L, 16)); C = torch.randn(1, L, 16)
    D = ssm.D.detach()
    y_ref = ref_scan(u, delta, A, B, C, D)
    y = ssm._selective_scan(u, delta, A, B, C, D)
    err = (y - y_ref).abs().max().item()
    print(f"L={L:5d}  ref|max|={y_ref.abs().max():8.3f}  orig|max|={y.abs().max():10.3e}  maxerr={err:.3e}  nan={torch.isnan(y).any().item()}")
