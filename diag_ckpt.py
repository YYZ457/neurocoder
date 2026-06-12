import torch
import sys; sys.path.insert(0,'.')

ckpt = torch.load('checkpoints/checkpoint-2500.pt', map_location='cpu', weights_only=False)

cfg = ckpt['config']
stats = ckpt['stats']

print(f"""Checkpoint config:
  d_model={cfg.d_model}, n_blocks={cfg.n_blocks}, n_experts={cfg.n_experts}
  expert_dim={cfg.expert_dim}, ssm_state_dim={cfg.ssm_state_dim}
  max_steps={cfg.max_steps}, warmup_steps={cfg.warmup_steps}
  lr={cfg.learning_rate}, batch_size={cfg.batch_size}

Stats: step={stats['step']}, total_tokens={stats['total_tokens']}
""")

sch = ckpt['scheduler_state_dict']
print(f'Scheduler last_epoch={sch.get("last_epoch", "?")}')

opt = ckpt['optimizer_state_dict']
for i, pg in enumerate(opt['param_groups']):
    print(f'  param_group[{i}]: lr={pg["lr"]:.6e}')
