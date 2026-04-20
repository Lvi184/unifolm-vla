# Copyright 2025 NVIDIA Corp. and affiliates. All rights reserved.
# Modified by [Junqiu YU/ Fudan University] in [2025].
# Modification: [rm and add some connect adapter to match with starVLA, e.g., "rm "].
# Action repeat is inspired by CogACT



from dataclasses import dataclass, field
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature
from unifolm_vla.model.modules.action_model.flow_matching_modules.action_encoder import (
    SinusoidalPositionalEncoding,
    swish,
)

from unifolm_vla.model.modules.action_model.flow_matching_modules.cross_attention_dit import DiT
from unifolm_vla.rlds_dataloader.constants import ACTION_DIM, PROPRIO_DIM, NUM_ACTIONS_CHUNK


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.layer2(F.relu(self.layer1(x)))


class ActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.layer1 = nn.Linear(action_dim, hidden_size)
        self.layer2 = nn.Linear(2 * hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        returns:   shape (B, T, hidden_size)
        """
        if actions.ndim == 2:
            # When action_horizon = 1, collate_fn squeezes out the T dimension -> (B, D)
            # Add it back manually
            B, D = actions.shape
            actions = actions.unsqueeze(1)
            T = 1
        else:
            B, T, _ = actions.shape
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError(
                "Expected `timesteps` to have shape (B,) so we can replicate across T."
            )

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.layer1(actions)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then layer2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.layer2(x))

        # 5) Finally W3 => (B, T, w)
        x = self.layer3(x)
        return x


@dataclass
class FlowmatchingActionHeadConfig(PretrainedConfig):
    """NOTE: N1.5 uses XEmbFlowmatchingPolicyHeadConfig as action head"""

    add_pos_embed: bool = field(
        default=True, metadata={"help": "Whether to add positional embedding"}
    )
    diffusion_model_cfg: dict = field(
        default=None, metadata={"help": "Diffusion model configuration."}
    )
    input_embedding_dim: int = field(
        default=1536, metadata={"help": "Input embedding channel dimension."}
    )

    hidden_size: int = field(default=1024, metadata={"help": "Input embedding dimension."})
    max_seq_len: int = field(default=1024, metadata={"help": "Maxium Sequence Length"})
    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})
    noise_beta_alpha: float = field(default=1.5, metadata={"help": ""})
    noise_beta_beta: float = field(default=1.0, metadata={"help": ""})
    noise_s: float = field(
        default=0.999, metadata={"help": "Flow matching noise Beta distribution s."}
    )
    num_timestep_buckets: int = field(
        default=1000, metadata={"help": "Number of timestep discretization buckets."}
    )
    num_inference_timesteps: int = field(
        default=None,
        metadata={"help": "Number of inference steps for noise diffusion."},
    )
    max_num_embodiments: int = field(default=32, metadata={"help": "Number of embodiments."})
    tune_projector: bool = field(default=True, metadata={"help": "Whether to tune the projector."})
    tune_diffusion_model: bool = field(
        default=True, metadata={"help": "Whether to tune the diffusion model."}
    )
    load_pretrained_det_decode_layer_path: str = field(
        default=None, metadata={"help": "Path to pretrained detection model."}
    )
    detection_coeff: float = field(default=1.0, metadata={"help": "Detection coefficient."})

    freeze_decode_layer: bool = field(default=False)
    expand_batch: int = field(default=None)
    use_vlln: bool = field(default=True)

    vl_self_attention_cfg: dict = field(default=None)
    num_target_vision_tokens: int = field(
        default=32, metadata={"help": "Number of target vision tokens."}
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


DiTConfig = {
    "DiT-B": {"input_embedding_dim": 768, "attention_head_dim": 64, "num_attention_heads": 12},
    "DiT-L": {"input_embedding_dim": 1536, "attention_head_dim": 48, "num_attention_heads": 32},
}

class FlowmatchingActionHead(nn.Module):
    def __init__(
        self,
        full_config,
    ):
        super().__init__()
        config = full_config.framework.action_model
        self.hidden_size = config.hidden_size
        self.full_config = full_config
        self.input_embedding_dim = config.input_embedding_dim
        diffusion_model_cfg = config.diffusion_model_cfg

        self.model = DiT(**diffusion_model_cfg)

        self.num_inference_timesteps = config.num_inference_timesteps

        # Force reload constants to pick up correct dimensions for current run (avoids cached imports)
        from importlib import reload
        import unifolm_vla.rlds_dataloader.constants
        reload(unifolm_vla.rlds_dataloader.constants)
        from unifolm_vla.rlds_dataloader.constants import ACTION_DIM, PROPRIO_DIM, NUM_ACTIONS_CHUNK

        self.action_dim = ACTION_DIM
        self.proprio_dim = PROPRIO_DIM
        self.action_horizon = NUM_ACTIONS_CHUNK

        self.state_encoder = MLP(
            input_dim=self.proprio_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )

        self.action_encoder = ActionEncoder(
            action_dim=self.action_dim,
            hidden_size=self.input_embedding_dim,
        )
        self.action_decoder = MLP(
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )
        self.future_tokens = nn.Embedding(config.num_target_vision_tokens, self.input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        self.config = config

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)


    def forward(self, vl_embs: torch.Tensor, actions: torch.Tensor, state: torch.Tensor = None):
        """
        vl_embs: shape (B, seq_length, feature_dim)
        actions: shape (B, future_action_window_size, D_action)
        """
        device = vl_embs.device

        # Embed noised action trajectory.
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]
        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized)


        # embed state
        state_features = self.state_encoder(state) if state is not None else None
        if state_features is not None:
            # Add sequence length dimension: [batch, hidden] -> [batch, 1, hidden]
            # Needed for concatenation along sequence dimension (dim 1)
            # Ensure we have correct number of dimensions
            state_features = state_features.reshape(-1, self.input_embedding_dim).unsqueeze(1)

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # state and action embedding along sequence dimension.
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
        
        # Debug print shapes
        # print(f"DEBUG DiT: vl_embs.shape[0] = {vl_embs.shape[0]}")
        # if state_features is not None:
        #     print(f"DEBUG DiT: state_features.shape = {tuple(state_features.shape)}")
        # print(f"DEBUG DiT: future_tokens.shape = {tuple(future_tokens.shape)}")
        # print(f"DEBUG DiT: actions.shape = {tuple(actions.shape)}")
        # print(f"DEBUG DiT: action_features.shape = {tuple(action_features.shape)}")
        
        sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1) \
            if state_features is not None else torch.cat((future_tokens, action_features), dim=1)
        # Join VLM features with state and action embedding along sequence dimension.
        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            timestep=t_discretized,
        )
        pred = self.action_decoder(model_output)
        pred_actions = pred[:, -actions.shape[1] :]

        # Slice out only the action portion of pred and target.
        loss = ((pred_actions - velocity) ** 2).mean()
        return loss

    @torch.no_grad()
    def predict_action(self, vl_embs: torch.Tensor, state: torch.Tensor = None) -> torch.Tensor:
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        actions = torch.randn(
            size=(batch_size, self.action_horizon, self.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        if state is not None:
            # Squeeze extra dimensions
            state = state.squeeze()
            
            # Ensure that the last dimension is proprio_dim - flatten everything else to batch
            if state.shape[-1] != self.proprio_dim:
                # Try to reshape - flatten all except last dim to batch
                total = state.numel() // self.proprio_dim
                state = state.reshape(total, self.proprio_dim)
            
            # If we got 1D tensor (just proprio - add batch dimension at front)
            if state.ndim == 1:
                state = state.unsqueeze(0)
            
            # Now shape is (batch_size, proprio_dim)
            state_features = self.state_encoder(state)  # DO NOT extra unsqueeze here - we'll normalize later
        else:
            state_features = None

        # Run denoising steps.
        for t in range(num_steps):
            t_cont = t / float(num_steps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory.
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )

            action_features = self.action_encoder(actions, timesteps_tensor)
            # Maybe add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # Join vision, language, state and action embedding along sequence dimension.
            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)

            # =============================================================
            #  Normalize all tensors to [B, N, D] consistently with forward()
            # =============================================================
            def _to_btn(x, name: str):
                """Normalize tensor to [B, N, D] format."""
                if x is None:
                    return None
                if x.ndim == 2:
                    # [B, D] -> [B, 1, D]
                    x = x.unsqueeze(1)
                elif x.ndim == 3:
                    # already [B, N, D] - do nothing
                    pass
                elif x.ndim == 4 and x.shape[1] == 1:
                    # [B, 1, N, D] -> [B, N, D]
                    x = x.squeeze(1)
                else:
                    raise RuntimeError(f"{name} has unsupported shape: {tuple(x.shape)}")
                return x

            state_features = _to_btn(state_features, "state_features")
            future_tokens = _to_btn(future_tokens, "future_tokens")
            action_features = _to_btn(action_features, "action_features")

            # Find consistent hidden dimension from available embeddings
            hidden_dim = None
            for t in (state_features, action_features, future_tokens):
                if t is not None:
                    hidden_dim = t.shape[-1]
                    break

            if hidden_dim is None:
                raise RuntimeError("All of state_features, future_tokens, action_features are None.")

            # Minimal fallback for shape mismatches (common from old cached code)
            if future_tokens is not None and future_tokens.shape[-1] != hidden_dim:
                if future_tokens.shape[-1] == 1:
                    future_tokens = future_tokens.expand(
                        future_tokens.shape[0],
                        future_tokens.shape[1],
                        hidden_dim,
                    )
                else:
                    raise RuntimeError(
                        f"future_tokens hidden dim mismatch: got {future_tokens.shape[-1]}, expected {hidden_dim}"
                    )

            # Final sanity checks
            tensors = [t for t in (state_features, future_tokens, action_features) if t is not None]
            if len(tensors) == 0:
                raise RuntimeError("No tensors available for concatenation in predict_action().")

            batch_size = tensors[0].shape[0]
            for name, t in [
                ("state_features", state_features),
                ("future_tokens", future_tokens),
                ("action_features", action_features),
            ]:
                if t is None:
                    continue
                if t.shape[0] != batch_size:
                    raise RuntimeError(
                        f"{name} batch mismatch: got {t.shape[0]}, expected {batch_size}"
                    )
                if t.shape[-1] != hidden_dim:
                    raise RuntimeError(
                        f"{name} hidden dim mismatch: got {t.shape[-1]}, expected {hidden_dim}"
                    )

            sa_embs = torch.cat(tensors, dim=1)

            # Run model forward.
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=timesteps_tensor,
            )
            pred = self.action_decoder(model_output)

            pred_velocity = pred[:, -self.action_horizon :]

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity
        return actions

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype



def get_action_model(config=None):
    """
    Factory: build FlowmatchingActionHead from global framework config.

    Args:
        config: Global config (expects config.framework.action_model namespace).

    Returns:
        FlowmatchingActionHead: Initialized FlowMatchingActionHead.
    """
    return FlowmatchingActionHead(
        full_config=config
    )
