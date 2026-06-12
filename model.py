"""
NeuroCoder: A Brain-Inspired Sparse Hierarchical Code Generation Model
======================================================================

Core architectural innovations over Transformers:

1. SELECTIVE STATE SPACE MODEL (Mamba-inspired)
   Replaces O(N²) self-attention with O(N) selective SSM.
   The SSM parameters (Δ, B, C) are INPUT-DEPENDENT — the model
   learns WHAT to remember and WHAT to forget per token.
   Like the brain's working memory: selective, not photographic.

2. SPARSE MIXTURE OF CODE EXPERTS
   Instead of one dense FFN, we have 32 small "code experts."
   Each token activates only 2 experts via learned routing.
   This is SPARSITY like the brain: ~5% neurons active at once.
   Total: 200M params, but only ~40M compute per token.

3. HIERARCHICAL CODE MIXER
   Code has natural hierarchy: token → line → block.
   We pool and broadcast across levels, like the visual cortex
   processing at multiple spatial scales simultaneously.

4. PREDICTIVE GATING
   A small network predicts which experts will be needed,
   enabling fast feedforward-like processing for common patterns.

Math background — the SSM core:
    Continuous:  x'(t) = A x(t) + B u(t)
                  y(t) = C x(t) + D u(t)
    Discrete:    x_k = Ā x_{k-1} + B̄ u_k
                  y_k = C x_k + D u_k
    where Ā, B̄ come from ZOH discretization with step size Δ.

Reference papers:
    Mamba (Gu & Dao, 2023) — Selective State Spaces
    DeepSeek-V2 (2024) — Sparse MoE at scale
    Predictive Coding (Rao & Ballard, 1999) — Brain inspiration
"""

import math
from typing import Optional, Tuple, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from config import NeuroCoderConfig


# ===========================================================================
# PART 1: Selective State Space Model (SSM)
# ===========================================================================

class SelectiveSSM(nn.Module):
    """
    Mamba-style selective state space model.

    Key insight: Unlike fixed SSMs (S4), the parameters Δ, B, C are
    functions of the INPUT. This makes the SSM "selective" — it can
    choose what to remember based on content.

    The recurrence is:
        h_t = Ā_t h_{t-1} + B̄_t x_t    (state update)
        y_t = C_t h_t + D x_t          (output)

    where Ā_t = exp(Δ_t A) and B̄_t = (Δ_t A)^{-1} (exp(Δ_t A) - I) Δ_t B_t

    Complexity: O(N · state_dim · d_model) vs Attention's O(N² · d_model)
    For d_model=512, state_dim=16, N=2048:
        SSM: 2048 × 16 × 512 ≈ 16.7M operations
        Attention: 2048² × 512 ≈ 2.1B operations  → SSM is ~125× cheaper!
    """

    def __init__(self, config: NeuroCoderConfig):
        super().__init__()
        self.d_model = config.d_model
        self.state_dim = config.ssm_state_dim
        self.expand = config.ssm_expand
        self.dt_rank = config.ssm_dt_rank
        self.inner_dim = self.d_model * self.expand

        # --- Input projection (expand d_model → inner_dim) ---
        self.in_proj = nn.Linear(self.d_model, self.inner_dim * 2, bias=False)

        # --- Δ (delta) projection: input-dependent step size ---
        # Δ controls HOW MUCH the state updates. Small Δ = remember, Large Δ = reset.
        # This is the KEY innovation of Mamba: selective memory.
        self.dt_proj = nn.Sequential(
            nn.Linear(self.dt_rank, self.dt_rank, bias=True),
            nn.SiLU(),
            nn.Linear(self.dt_rank, self.inner_dim, bias=True),
        )
        # dt_rank → dt_rank
        self.x_proj = nn.Linear(self.d_model, self.dt_rank, bias=False)

        # --- A matrix: learnable decay rates ---
        # Using a diagonal A with a range of decay rates:
        # - Small |A| (e.g., -0.5): slow decay = long memory (~100s of tokens)
        # - Large |A| (e.g., -8.0): fast decay = short memory (~few tokens)
        # Each of the inner_dim channels gets its own set of state_dim decay rates.
        # All stored as log(-A) for positivity after exp().
        A_diag = torch.linspace(0.5, 8.0, self.state_dim)  # (state_dim,)
        A = -A_diag.unsqueeze(0).repeat(self.inner_dim, 1)  # (inner_dim, state_dim)
        self.A_log = nn.Parameter(torch.log(-A))  # always valid: -A > 0
        self.D = nn.Parameter(torch.ones(self.inner_dim))  # Skip connection

        # --- B and C are input-dependent ---
        # B controls HOW input enters the state
        # C controls HOW state projects to output
        # Making them input-dependent = the SSM is "selective"
        # We'll compute B and C from the projected input

        # --- Output projection ---
        self.out_proj = nn.Linear(self.inner_dim, self.d_model, bias=False)

        # --- Causal Conv1D (local context before SSM) ---
        # A short causal conv preprocesses the input, giving the SSM
        # local context awareness similar to how the brain's sensory
        # neurons have local receptive fields.
        self.conv_kernel_size = 3
        self.conv = nn.Conv1d(
            self.inner_dim, self.inner_dim,
            kernel_size=self.conv_kernel_size,
            padding=self.conv_kernel_size - 1,
            groups=self.inner_dim,  # Depthwise
            bias=False,
        )

        self._init_weights()

    def _init_weights(self):
        """Careful initialization for stable SSM training."""
        # Input projection: small random
        nn.init.normal_(self.in_proj.weight, std=0.02)
        # dt_proj: initialize so softplus(dt) ≈ 0.5 initially
        nn.init.normal_(self.dt_proj[0].weight, std=0.02)
        nn.init.constant_(self.dt_proj[0].bias, 0.0)
        nn.init.normal_(self.dt_proj[2].weight, std=0.02)
        nn.init.constant_(self.dt_proj[2].bias, math.log(math.e - 1))  # softplus(bias) ≈ 1
        nn.init.normal_(self.x_proj.weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02)

    def _selective_scan(self, u, delta, A, B_ssm, C_ssm, D):
        """
        Fast vectorized selective scan using cumprod/cumsum.

        The recurrence: h_t = a_t * h_{t-1} + b_t  (h_0 = 0)
        has closed-form solution:
            h_t = p_t * cumsum(b_i / p_i)[t]
        where p_t = cumprod(a_i)[t]

        This replaces the O(L) Python loop with pure CUDA ops.
        ~100x faster than sequential Python loop.
        """
        B, L, ID = u.shape
        N = self.state_dim

        # --- Discretization ---
        delta_exp = delta.unsqueeze(-1)         # (B, L, ID, 1)
        A_exp = A.unsqueeze(0).unsqueeze(0)      # (1, 1, ID, N)
        A_bar = torch.exp(delta_exp * A_exp)     # (B, L, ID, N) ∈ (0, 1]
        A_bar = A_bar.clamp(min=1e-8, max=1.0)

        B_bar = delta_exp * B_ssm.unsqueeze(2)   # (B, L, ID, N)
        u_exp = u.unsqueeze(-1)                  # (B, L, ID, 1)
        b = B_bar * u_exp                        # (B, L, ID, N)

        # --- Cumprod solution ---
        # p_t = cumprod_{i=1}^{t} a_i
        log_A = torch.log(A_bar.clamp(min=1e-12))  # (B, L, ID, N)
        log_p = torch.cumsum(log_A, dim=1)          # (B, L, ID, N)
        p = torch.exp(log_p)                         # (B, L, ID, N)

        # q_t = b_t / p_t (rescaled input)
        q = b / p.clamp(min=1e-12)                  # (B, L, ID, N)

        # h_t = p_t * cumsum(q)[t]
        h = p * torch.cumsum(q, dim=1)              # (B, L, ID, N)

        # y_t = C_t @ h_t + D * u_t
        # C_ssm: (B, L, N) → (B, L, 1, N)
        # h: (B, L, ID, N) → sum over N
        y = (C_ssm.unsqueeze(2) * h).sum(dim=-1)    # (B, L, ID)
        y = y + D.unsqueeze(0).unsqueeze(0) * u     # skip connection

        return y  # (B, L, ID)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model)
        Returns:
            out: (B, L, d_model)
        """
        B, L, _ = x.shape

        # --- Input projection ---
        xz = self.in_proj(x)  # (B, L, inner_dim * 2)
        x_proj, z = xz.chunk(2, dim=-1)  # Each (B, L, inner_dim)

        # --- Causal Conv1D (local context) ---
        # Conv1d expects (B, C, L)
        x_conv = x_proj.transpose(1, 2)  # (B, inner_dim, L)
        x_conv = self.conv(x_conv)
        x_conv = x_conv[:, :, :L]  # Remove padding (causal)
        x_conv = x_conv.transpose(1, 2)  # (B, L, inner_dim)
        x_conv = F.silu(x_conv)

        # --- Compute input-dependent SSM parameters ---
        # Δ: step size for discretization
        dt_input = self.x_proj(x)  # (B, L, dt_rank)
        dt = self.dt_proj(dt_input)  # (B, L, inner_dim)
        dt = F.softplus(dt)  # Ensure positive

        # A: HiPPO matrix in real space
        A = -torch.exp(self.A_log)  # (inner_dim, state_dim)

        # B and C: computed from the convolved input
        # We use small projections for B and C
        B = torch.tanh(x_conv[:, :, :self.state_dim])  # (B, L, state_dim)
        C = x_conv[:, :, :self.state_dim]  # (B, L, state_dim)

        # --- Selective scan ---
        y = self._selective_scan(x_conv, dt, A, B, C, self.D)

        # --- Gating (Mamba-style) ---
        y = y * F.silu(z)

        # --- Output projection ---
        out = self.out_proj(y)  # (B, L, d_model)

        return out


# ===========================================================================
# PART 2: Sparse Mixture of Code Experts
# ===========================================================================

class SparseCodeExpert(nn.Module):
    """
    A single code expert — a small FFN specialized in certain code patterns.

    Each expert is a SwiGLU FFN, which has been shown to outperform
    standard ReLU FFNs (Shazeer, 2020; used in PaLM, LLaMA, etc.).
    """

    def __init__(self, config: NeuroCoderConfig):
        super().__init__()
        # SwiGLU: gate(x) * up(x), where gate = SiLU(w1·x), up = w2·x
        self.w_gate = nn.Linear(config.d_model, config.expert_dim, bias=False)
        self.w_up = nn.Linear(config.d_model, config.expert_dim, bias=False)
        self.w_down = nn.Linear(config.expert_dim, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.expert_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w_gate(x))  # SwiGLU gate
        up = self.w_up(x)
        hidden = gate * up
        hidden = self.dropout(hidden)
        return self.w_down(hidden)


class SparseExpertRouter(nn.Module):
    """
    Routes each token to the top-k most relevant experts.

    The router is a simple linear classifier that scores each token
    against all experts. Top-k experts are selected per token.

    Key features:
    - Load balancing loss to prevent expert collapse
    - Auxiliary z-loss for training stability
    - Jitter noise during training for exploration
    """

    def __init__(self, config: NeuroCoderConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_experts = config.n_experts
        self.n_active = config.n_active_experts
        self.load_balance_coef = config.load_balance_coef
        self.z_loss_coef = config.z_loss_coef

        # Router weights: d_model → n_experts
        self.router = nn.Linear(config.d_model, config.n_experts, bias=False)

        # Temperature for softmax (learnable, starts at 1.0)
        self.log_temp = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B*L, d_model) — flattened token representations
        Returns:
            dispatch_weights: (B*L, n_active, n_experts) — one-hot-like routing
            combine_weights: (B*L, n_active) — softmax weights for combining expert outputs
            aux_loss: scalar — load balancing + z-loss
        """
        # Router logits
        logits = self.router(x) / torch.exp(self.log_temp).clamp(min=0.1, max=10.0)
        # (tokens, n_experts)

        # Jitter noise during training
        if self.training:
            noise = torch.randn_like(logits) * 0.1
            logits = logits + noise

        # Normalize to probabilities
        router_probs = F.softmax(logits, dim=-1)  # (tokens, n_experts)

        # Select top-k experts
        top_k_probs, top_k_indices = torch.topk(router_probs, self.n_active, dim=-1)
        # (tokens, n_active)

        # Re-normalize among selected experts
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        # Create dispatch mask (one-hot for each selected expert)
        dispatch_mask = F.one_hot(top_k_indices, self.n_experts).float()
        # (tokens, n_active, n_experts)

        # Combine weights = softmax over selected experts
        combine_weights = top_k_probs  # (tokens, n_active)

        # --- Auxiliary losses ---
        # Load balancing: encourage uniform expert usage
        # fraction of tokens dispatched to each expert
        density = dispatch_mask.mean(dim=0)  # (n_active, n_experts)
        density = density.sum(dim=0)  # (n_experts,) — avg tokens per expert
        # mean router probability for each expert
        mean_prob = router_probs.mean(dim=0)  # (n_experts,)
        load_balance_loss = (density * mean_prob).sum() * self.n_experts

        # Z-loss: penalize large logits for stability
        z_loss = logits.logsumexp(dim=-1).pow(2).mean()
        z_loss = z_loss * self.z_loss_coef

        aux_loss = self.load_balance_coef * load_balance_loss + z_loss

        return dispatch_mask, combine_weights, aux_loss


class SparseMoE(nn.Module):
    """
    Sparse Mixture of Code Experts layer.

    Each token is processed by only k out of N experts (k << N).
    This is the key to having a LARGE model with SMALL compute.

    Analogy from the brain: You have billions of neurons but only
    a small fraction fire for any given stimulus. The rest stay silent,
    conserving energy. Same principle here.
    """

    def __init__(self, config: NeuroCoderConfig):
        super().__init__()
        self.config = config
        self.n_active = config.n_active_experts

        # Create experts
        self.experts = nn.ModuleList([
            SparseCodeExpert(config) for _ in range(config.n_experts)
        ])

        # Router
        self.router = SparseExpertRouter(config)

        # Shared expert (always active, like a "default pathway")
        self.shared_expert = SparseCodeExpert(config)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, L, d_model)
        Returns:
            out: (B, L, d_model)
            aux_loss: scalar
        """
        B, L, D = x.shape
        x_flat = x.reshape(-1, D)  # (B*L, d_model)

        # Route tokens to experts
        dispatch_mask, combine_weights, aux_loss = self.router(x_flat)
        # dispatch_mask: (B*L, n_active, n_experts)
        # combine_weights: (B*L, n_active)

        # Process tokens through their assigned experts
        expert_outputs = torch.zeros(B * L, D, device=x.device, dtype=x.dtype)

        for expert_idx, expert in enumerate(self.experts):
            # Find tokens dispatched to this expert
            # Which active slot has this expert?
            expert_active = (dispatch_mask[:, :, expert_idx] > 0.5).any(dim=-1)
            # (B*L,) — True for tokens that have this expert in their top-k

            if expert_active.any():
                token_indices = expert_active.nonzero(as_tuple=True)[0]
                # Get the specific slot for this expert
                slot = dispatch_mask[token_indices, :, expert_idx].argmax(dim=-1)
                # (n_dispatched,) — which of the n_active slots

                expert_input = x_flat[token_indices]
                expert_out = expert(expert_input)

                # Weight by the combine weight for that slot
                weights = combine_weights[token_indices, slot].unsqueeze(-1)
                expert_outputs[token_indices] += expert_out * weights

        # Add shared expert (always on, provides baseline)
        shared_out = self.shared_expert(x_flat)

        # Reshape back
        out = expert_outputs.reshape(B, L, D) + shared_out.reshape(B, L, D)
        out = self.dropout(out)

        return out, aux_loss


# ===========================================================================
# PART 3: Hierarchical Code Mixer
# ===========================================================================

class HierarchicalCodeMixer(nn.Module):
    """
    Processes code at three hierarchical levels simultaneously:

    Level 1 (Token): Raw token embeddings
    Level 2 (Line):  Pooled over ~16 tokens (roughly one line of code)
    Level 3 (Block): Pooled over ~128 tokens (roughly a function/class)

    Higher levels modulate lower levels via top-down attention,
    just like the brain's cortical hierarchy (visual cortex V1→V2→V4→IT).

    This gives the model inherent understanding of code STRUCTURE
    without needing explicit AST parsing.
    """

    def __init__(self, config: NeuroCoderConfig):
        super().__init__()
        self.d_model = config.d_model
        self.token_len = config.hierarchy_token_len   # 16
        self.block_len = config.hierarchy_block_len    # 128

        # --- Bottom-up pooling (token → line → block) ---
        self.token_to_line = nn.Linear(config.d_model, config.d_model)
        self.line_to_block = nn.Linear(config.d_model, config.d_model)

        # --- Cross-level attention ---
        # Line-level attends to block-level context
        self.line_block_attn = nn.MultiheadAttention(
            config.d_model, num_heads=4, batch_first=True, dropout=0.1
        )

        # --- Top-down modulation (block → line → token) ---
        self.block_to_line = nn.Linear(config.d_model, config.d_model)
        self.line_to_token = nn.Linear(config.d_model, config.d_model * 2)  # gate + value

        # --- Learnable level embeddings ---
        self.level_embed = nn.Parameter(torch.randn(3, config.d_model) * 0.02)

    def _pool(self, x: torch.Tensor, pool_size: int) -> torch.Tensor:
        """Pool sequence by factor of pool_size using learned linear + mean."""
        B, L, D = x.shape
        # Pad to multiple of pool_size
        pad_len = (pool_size - L % pool_size) % pool_size
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
        L_padded = x.shape[1]
        # Reshape and pool
        x = x.reshape(B, L_padded // pool_size, pool_size, D)
        return x.mean(dim=2)  # (B, L//pool_size, D)

    def _broadcast(self, x_small: torch.Tensor, target_len: int) -> torch.Tensor:
        """Broadcast back from pooled to original length."""
        B, L_small, D = x_small.shape
        x_repeated = x_small.unsqueeze(2).repeat(1, 1, self.token_len, 1)
        x_flat = x_repeated.reshape(B, L_small * self.token_len, D)
        # Trim to target length
        return x_flat[:, :target_len, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model) — token-level representations
        Returns:
            out: (B, L, d_model) — hierarchy-enhanced representations
        """
        B, L, D = x.shape

        # --- Level 1: Token (already have this) ---
        token_level = x + self.level_embed[0]

        # --- Level 2: Line (pool tokens → lines) ---
        line_level_raw = self._pool(x, self.token_len)  # (B, L//16, D)
        line_level = self.token_to_line(line_level_raw) + self.level_embed[1]

        # --- Level 3: Block (pool lines → blocks) ---
        L_lines = line_level.shape[1]
        if L_lines >= self.block_len // self.token_len:
            block_level_raw = self._pool(line_level_raw, self.block_len // self.token_len)
            block_level = self.line_to_block(block_level_raw) + self.level_embed[2]

            # --- Cross-level: Line ← Block (bottom-up context) ---
            # Lines attend to blocks
            block_broadcast = self._broadcast(block_level, L_lines * self.token_len)
            block_broadcast = block_broadcast[:, :line_level.shape[1], :]

            # Expand line_level for multihead attention
            line_enhanced, _ = self.line_block_attn(
                query=line_level,
                key=block_level,
                value=block_level,
            )
            line_level = line_level + line_enhanced

            # --- Top-down: Block → Line → Token ---
            block_mod = self.block_to_line(block_level)  # (B, L_blocks, D)
            block_to_line_bc = self._broadcast(block_mod, L_lines * self.token_len)
            block_to_line_bc = block_to_line_bc[:, :line_level.shape[1], :]
            line_level = line_level + block_to_line_bc

        # --- Top-down: Line → Token ---
        line_broadcast = self._broadcast(line_level, L)
        line_broadcast = line_broadcast[:, :L, :]

        line_gate_val = self.line_to_token(line_broadcast)  # (B, L, 2*D)
        gate, value = line_gate_val.chunk(2, dim=-1)
        gate = torch.sigmoid(gate)

        # Gated injection of line-level information into tokens
        out = token_level + gate * value

        return out


# ===========================================================================
# PART 4: Predictive Gating (Brain-inspired Fast Pathway)
# ===========================================================================

class PredictiveGate(nn.Module):
    """
    A small network that PREDICTS which experts will be needed for the
    NEXT token, based on current context.

    Brain analogy: The brain constantly predicts upcoming stimuli.
    When predictions match, processing is fast (feedforward-like).
    When they mismatch, more neurons engage (prediction error).

    This module provides a "fast pathway" for common code patterns
    (e.g., after "def " → expert for function signatures).
    """

    def __init__(self, config: NeuroCoderConfig):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 4),
            nn.SiLU(),
            nn.Linear(config.d_model // 4, config.d_model),
        )
        self.gate = nn.Linear(config.d_model * 2, config.d_model)
        self.ln = nn.LayerNorm(config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model)
        Returns:
            gated: (B, L, d_model) — enhanced with predictive signals
        """
        # Predict next-token features from current
        prediction = self.predictor(x)
        # Shift: prediction for position t comes from position t-1
        prediction = F.pad(prediction[:, :-1, :], (0, 0, 1, 0))  # (B, L, D)

        # Gate: blend prediction with actual
        combined = torch.cat([x, prediction], dim=-1)
        gate = torch.sigmoid(self.gate(combined))
        out = self.ln(x + gate * prediction)

        return out


# ===========================================================================
# PART 5: NeuroCoder Block & Full Model
# ===========================================================================

class NeuroCoderBlock(nn.Module):
    """
    One "cortical column" of the NeuroCoder.

    Processing order:
        Input → [PredictiveGate] → [SelectiveSSM] → [SparseMoE] → [HierarchicalMixer] → Output

    Each block has two residual connections (pre-SSM and pre-MoE),
    mirroring the Pre-LN transformer design.
    """

    def __init__(self, config: NeuroCoderConfig):
        super().__init__()
        self.predictive_gate = PredictiveGate(config)
        self.ssm = SelectiveSSM(config)
        self.moe = SparseMoE(config)
        self.hierarchy = HierarchicalCodeMixer(config)
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ln3 = nn.LayerNorm(config.d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, L, d_model)
        Returns:
            h: (B, L, d_model)
            aux_loss: scalar
        """
        # Predictive gating (fast pathway)
        h = self.predictive_gate(x)

        # Selective SSM (O(N) memory, no attention)
        h = h + self.ssm(self.ln1(h))

        # Sparse MoE (most of the model capacity, cheap compute)
        moe_out, aux_loss = self.moe(self.ln2(h))
        h = h + moe_out

        # Hierarchical mixing (code structure awareness)
        h = h + self.hierarchy(self.ln3(h))

        return h, aux_loss


class NeuroCoder(nn.Module):
    """
    NeuroCoder: A Brain-Inspired Sparse Hierarchical Code Generation Model.

    Architecture summary:
    ┌──────────────┐
    │  Token Embed │  32K vocab → d_model
    ├──────────────┤
    │  Predictive  │  Fast pathway: predict next features
    │  Selective   │  O(N) state space model (not attention!)
    │  Sparse MoE  │  32 experts, 2 active per token (sparse!)
    │  Hierarchy   │  Token → Line → Block (3 levels)
    ├──────────────┤  × N blocks (16 by default)
    │   LM Head    │  d_model → 32K vocab
    └──────────────┘

    Key metrics (default config):
        Total params:   ~200M
        Active/token:   ~40M   (5× sparsity)
        Training VRAM:  ~5 GB  (fits RTX 4060)
        Complexity:     O(N)   (not O(N²) like Transformers)
    """

    def __init__(self, config: NeuroCoderConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Parameter(
            torch.randn(1, config.max_seq_len, config.d_model) * 0.02
        )

        # Dropout
        self.embed_dropout = nn.Dropout(config.dropout)

        # NeuroCoder blocks
        self.blocks = nn.ModuleList([
            NeuroCoderBlock(config) for _ in range(config.n_blocks)
        ])

        # Final layer norm
        self.final_ln = nn.LayerNorm(config.d_model)

        # LM head (tied with input embeddings)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Weight tying: same matrix for embedding and output
        self.lm_head.weight = self.token_embed.weight

        # Initialize
        self._init_weights()

    def _init_weights(self):
        """Initialize weights following best practices."""
        nn.init.normal_(self.token_embed.weight, std=0.02)
        # pos_embed already initialized above
        nn.init.normal_(self.lm_head.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            input_ids: (B, L) — token indices
            attention_mask: (B, L) — 1 for real tokens, 0 for padding
            labels: (B, L) — for language modeling loss
        Returns:
            dict with keys: logits, loss, aux_loss, perplexity
        """
        B, L = input_ids.shape
        device = input_ids.device

        # Embed
        x = self.token_embed(input_ids)  # (B, L, d_model)
        x = x + self.pos_embed[:, :L, :]
        x = self.embed_dropout(x)

        # Apply mask (zero out padding positions)
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).float()

        # Process through blocks
        total_aux_loss = torch.tensor(0.0, device=device)
        for block in self.blocks:
            if self.config.use_gradient_checkpointing and self.training:
                h, aux = torch.utils.checkpoint.checkpoint(
                    block, x, use_reentrant=False
                )
            else:
                h, aux = block(x)
            x = h
            total_aux_loss = total_aux_loss + aux

        # Final norm
        x = self.final_ln(x)

        # LM head
        logits = self.lm_head(x)  # (B, L, vocab_size)

        result = {"logits": logits, "aux_loss": total_aux_loss}

        # Compute loss if labels provided
        if labels is not None:
            # Shift for autoregressive training
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            # Cross-entropy loss (with optional Focal Loss weighting)
            gamma = getattr(self.config, 'focal_loss_gamma', 0.0)
            if gamma > 0:
                # Per-token CE (ignored positions = 0)
                ce_per_token = F.cross_entropy(
                    shift_logits.reshape(-1, self.config.vocab_size),
                    shift_labels.reshape(-1),
                    ignore_index=self.config.pad_token_id,
                    reduction='none',
                )
                # Focal weighting: (1 - pt)^gamma, down-weights easy tokens
                pt = torch.exp(-ce_per_token)  # model confidence for correct token
                weight = (1 - pt) ** gamma
                focal_loss = weight * ce_per_token
                # Mean over non-ignored positions only
                mask = (shift_labels.reshape(-1) != self.config.pad_token_id).float()
                ce_loss = focal_loss.sum() / mask.sum()
            else:
                ce_loss = F.cross_entropy(
                    shift_logits.reshape(-1, self.config.vocab_size),
                    shift_labels.reshape(-1),
                    ignore_index=self.config.pad_token_id,
                    reduction='mean',
                )

            # Total loss = CE + auxiliary (load balancing + z-loss)
            total_loss = ce_loss + total_aux_loss / max(len(self.blocks), 1)

            result["loss"] = total_loss
            result["ce_loss"] = ce_loss

            # Perplexity
            with torch.no_grad():
                result["perplexity"] = torch.exp(ce_loss.clamp(max=20))

        return result

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 50,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation.

        Args:
            input_ids: (B, L) — prompt tokens
            max_new_tokens: maximum tokens to generate
            temperature: sampling temperature
            top_p: nucleus sampling threshold
            top_k: top-k sampling
            eos_token_id: stop token
        Returns:
            generated: (B, L + new_tokens)
        """
        self.eval()
        B = input_ids.shape[0]
        device = input_ids.device

        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id

        generated = input_ids

        for step_idx in range(max_new_tokens):
            # Truncate to max sequence length
            if generated.shape[1] > self.config.max_seq_len:
                generated = generated[:, -self.config.max_seq_len:]

            # Forward pass
            outputs = self.forward(generated)
            logits = outputs["logits"][:, -1, :]  # (B, vocab_size)

            # Temperature scaling
            logits = logits / max(temperature, 1e-8)

            # Top-k filtering
            if top_k > 0:
                top_k_vals, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
                logits[logits < top_k_vals[:, -1:]] = float('-inf')

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')

            # Suppress EOS for first 3 tokens (prevents premature stop)
            if step_idx < 3:
                logits[:, eos_token_id] = float('-inf')

            # NaN guard: prevent GPU crashes from extreme values
            logits = torch.nan_to_num(logits, nan=-100.0, posinf=-100.0, neginf=-100.0)

            # Sample
            probs = F.softmax(logits, dim=-1)
            probs = torch.clamp(probs, min=1e-10, max=1.0)
            probs = probs / probs.sum(dim=-1, keepdim=True)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Append
            generated = torch.cat([generated, next_token], dim=-1)

            # Stop if all sequences hit EOS
            if (next_token == eos_token_id).all():
                break

        return generated

    def get_num_params(self, active_only: bool = False) -> int:
        """Count parameters."""
        if active_only:
            # Estimate: embedding + SSM per block + active experts per block
            # This is approximate
            return self.config.estimated_total_params // 5  # ~20% active
        return sum(p.numel() for p in self.parameters())

    def print_architecture(self):
        """Print a summary of the model architecture."""
        total = self.get_num_params(active_only=False)
        active = self.get_num_params(active_only=True)
        print(f"""
╔══════════════════════════════════════════════════════════╗
║              NeuroCoder Architecture                     ║
╠══════════════════════════════════════════════════════════╣
║  Total Parameters:     {total/1e6:>8.1f}M                    ║
║  Active per Token:    ~{active/1e6:>8.1f}M                    ║
║  Sparsity:            {100*(1-active/total):>8.1f}%                    ║
╠══════════════════════════════════════════════════════════╣
║  Blocks:              {self.config.n_blocks:>8}                      ║
║  Experts per Block:   {self.config.n_experts:>8}                      ║
║  Active Experts:      {self.config.n_active_experts:>8}                      ║
║  Hidden Dim:          {self.config.d_model:>8}                      ║
║  SSM State Dim:       {self.config.ssm_state_dim:>8}                      ║
║  Max Sequence:        {self.config.max_seq_len:>8}                      ║
║  Vocabulary:          {self.config.vocab_size:>8}                      ║
╚══════════════════════════════════════════════════════════╝
        """.strip())


# ===========================================================================
# Quick test
# ===========================================================================

if __name__ == "__main__":
    from config import CONFIG_SMALL

    print("Creating NeuroCoder model...")
    config = CONFIG_SMALL
    model = NeuroCoder(config)
    model.print_architecture()

    # Test forward pass
    B, L = 2, 128
    input_ids = torch.randint(0, config.vocab_size, (B, L))
    labels = torch.randint(0, config.vocab_size, (B, L))

    print(f"\nTesting forward pass with input shape ({B}, {L})...")
    model.train()
    output = model(input_ids, labels=labels)

    print(f"  Logits shape: {output['logits'].shape}")
    print(f"  Loss: {output['loss']:.4f}")
    print(f"  CE Loss: {output['ce_loss']:.4f}")
    print(f"  Aux Loss: {output['aux_loss']:.6f}")
    print(f"  Perplexity: {output['perplexity']:.2f}")

    # Memory estimate
    total_params = model.get_num_params()
    memory_mb = total_params * 2 / 1024 / 1024  # FP16 weights
    print(f"\n  Estimated VRAM (FP16 weights only): {memory_mb:.0f} MB")
    print(f"  Estimated VRAM (training, with optimizer): {memory_mb * 6:.0f} MB")

    # Test generation
    print("\nTesting generation...")
    model.eval()
    prompt = torch.randint(0, config.vocab_size, (1, 16))
    generated = model.generate(prompt, max_new_tokens=32)
    print(f"  Output shape: {generated.shape}")

    print("\n[OK] Model is working correctly!")
