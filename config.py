"""
NeuroCoder Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~
A brain-inspired sparse hierarchical model for Python code generation.
Designed for RTX 4060 (8 GB VRAM).

Architecture innovation: Selective SSM + Sparse MoE + Hierarchical Mixing
Total params: ~200M | Active per token: ~40M | Training memory: ~5 GB
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class NeuroCoderConfig:
    """Core architecture hyperparameters."""

    # Vocabulary
    vocab_size: int = 32768  # 32K code-optimized BPE tokens
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    # Model dimensions
    d_model: int = 512          # Hidden dimension (beams of the brain)
    n_blocks: int = 16          # NeuroCoder blocks (cortical columns)
    n_experts: int = 32         # Total experts per MoE layer
    n_active_experts: int = 2   # Top-k active experts (sparse!)
    expert_dim: int = 2048      # FFN dimension inside each expert

    # State Space Model (Mamba-style selective SSM)
    ssm_state_dim: int = 16     # Latent state dimension
    ssm_dt_rank: int = 64       # Rank for Δ projection
    ssm_expand: int = 2         # Expansion factor for SSM input projection

    # Hierarchical code structure
    n_hierarchy_levels: int = 3          # token, line, block
    hierarchy_token_len: int = 16        # ~1 line
    hierarchy_block_len: int = 128       # ~1 function/class

    # Regularization & stability
    dropout: float = 0.1
    expert_dropout: float = 0.1
    load_balance_coef: float = 0.01     # Aux loss weight for expert balance
    z_loss_coef: float = 0.001          # Router z-loss for stability

    # Training
    max_seq_len: int = 2048
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 1000
    max_steps: int = 100000
    batch_size: int = 2           # Small — fits 4060
    gradient_accumulation: int = 8  # Effective batch = 2 × 8 = 16
    grad_clip: float = 1.0

    # Precision & memory
    use_bf16: bool = True         # BF16 on 4060 for training
    use_gradient_checkpointing: bool = True
    use_flash_ssm: bool = False   # Pure PyTorch SSM for portability

    # Inference
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    max_new_tokens: int = 512

    @property
    def total_expert_params(self) -> int:
        """Total parameters across all experts in one MoE layer."""
        # Each expert: 2 linear layers × d_model → expert_dim → d_model
        per_expert = 2 * self.d_model * self.expert_dim
        return self.n_experts * per_expert

    @property
    def active_expert_params(self) -> int:
        """Parameters actually used per token in MoE."""
        per_expert = 2 * self.d_model * self.expert_dim
        return self.n_active_experts * per_expert

    @property
    def estimated_total_params(self) -> int:
        """Rough estimate of total model parameters."""
        # Embedding
        emb = self.vocab_size * self.d_model
        # SSM per block: input proj + A+B+C+D+dt_proj + output proj
        ssm = self.n_blocks * (
            2 * self.d_model * (self.d_model * self.ssm_expand)  # in/out proj
            + self.d_model * self.ssm_state_dim * 3               # B, C, D
            + self.ssm_dt_rank * self.d_model * 2                # dt proj
        )
        # MoE per block
        moe = self.n_blocks * self.total_expert_params
        # Router per block
        router = self.n_blocks * self.d_model * self.n_experts
        # Hierarchy mixer
        hier = self.n_blocks * self.d_model * self.n_hierarchy_levels * 2
        # LM head
        head = self.vocab_size * self.d_model
        return emb + ssm + moe + router + hier + head


@dataclass
class TrainingConfig:
    """Everything needed to launch training."""

    # Data
    train_data_path: str = ""
    eval_data_path: str = ""
    data_cache_dir: str = "D:/NeuroCoder/data_cache"

    # Output
    output_dir: str = "D:/NeuroCoder/checkpoints"
    log_dir: str = "D:/NeuroCoder/logs"
    save_every: int = 500
    eval_every: int = 500
    log_every: int = 10

    # Optimization
    optimizer: str = "adamw"       # adamw | lion | sophia
    lr_schedule: str = "cosine"    # cosine | linear | constant
    beta1: float = 0.9
    beta2: float = 0.95

    # Distributed (single GPU for 4060)
    device: str = "cuda"
    compile_model: bool = False    # torch.compile if PyTorch >= 2.0

    # Resume
    resume_from: Optional[str] = None


# Pre-built config for RTX 4060 (8 GB) — ~250M params, ~5GB VRAM training
CONFIG_4060 = NeuroCoderConfig(
    d_model=512,
    n_blocks=14,
    n_experts=24,
    n_active_experts=2,
    expert_dim=1536,
    ssm_state_dim=16,
    ssm_dt_rank=64,
    batch_size=2,
    gradient_accumulation=8,
    use_bf16=True,
    use_gradient_checkpointing=True,
)

# Smaller config for quick experiments (~193M params, ~3GB VRAM training)
CONFIG_SMALL = NeuroCoderConfig(
    d_model=320,
    n_blocks=10,
    n_experts=16,
    n_active_experts=2,
    expert_dim=1024,
    ssm_state_dim=16,
    ssm_dt_rank=48,
    batch_size=2,
    gradient_accumulation=8,
)

# Cloud config: optimized for 48GB AMD GPU (~1.37B total, ~172M active)
CONFIG_CLOUD = NeuroCoderConfig(
    d_model=640,
    n_blocks=14,
    n_experts=24,
    n_active_experts=2,
    expert_dim=2048,
    ssm_state_dim=24,
    ssm_dt_rank=64,
    batch_size=4,
    gradient_accumulation=8,
)

# Aggressive config — pushes 4060 to its limit (~350M params)
CONFIG_MAX = NeuroCoderConfig(
    d_model=640,
    n_blocks=20,
    n_experts=48,
    n_active_experts=3,
    expert_dim=2560,
    ssm_state_dim=24,
    ssm_dt_rank=80,
    batch_size=1,
    gradient_accumulation=16,
    use_gradient_checkpointing=True,
)
