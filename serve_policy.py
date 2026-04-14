"""
UnifoLM-VLA serve_policy.py (OpenPI compatible)
=============================================

Usage (same format as ACoT-VLA):
    python serve_policy.py \
      --env G2SIM \
      --port 8999 \
      --policy checkpoint \
      --policy.dir=./results/unifolm_vla_agibot_v1/checkpoints/steps_8000_pytorch_model.pt
"""
import dataclasses
import enum
import logging
import socket
import tyro

# Import our UnifoLM-VLA openpi policy
from unifolm_openpi_policy import UnifoLMOpenPIPolicy

try:
    from openpi.serving import websocket_policy_server
    HAS_OPENPI = True
except ImportError:
    HAS_OPENPI = False
    logging.warning("OpenPI not found, will use FastAPI fallback")


class EnvMode(enum.Enum):
    """Supported environments (for compatibility with ACoT-VLA)."""
    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"
    VLABENCH = "vlabench"
    LIBEROPLUS = "liberoplus"
    G2SIM = "g2sim"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""
    config: str = "unifolm_vla"
    dir: str = "./results/unifolm_vla_agibot_v1/checkpoints/steps_8000_pytorch_model.pt"


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script (ACoT-VLA compatible)."""
    env: EnvMode = EnvMode.G2SIM
    default_prompt: str | None = None
    port: int = 8999
    record: bool = False
    
    # UnifoLM-VLA specific
    vlm_pretrained_path: str = "/root/gpufree-data/unifolm-weights/UnifoLM-VLM-Base"
    unnorm_key: str = "rlds_dataset"
    center_crop: bool = False
    use_bf16: bool = True
    
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.G2SIM: Checkpoint(
        config="unifolm_vla",
        dir="./results/unifolm_vla_agibot_v1/checkpoints/steps_8000_pytorch_model.pt",
    )
}


def create_policy(args: Args) -> UnifoLMOpenPIPolicy:
    """Create a policy from the given arguments."""
    if isinstance(args.policy, Checkpoint):
        ckpt_path = args.policy.dir
    elif isinstance(args.policy, Default):
        if checkpoint := DEFAULT_CHECKPOINT.get(args.env):
            ckpt_path = checkpoint.dir
        else:
            raise ValueError(f"No default checkpoint for env: {args.env}")
    else:
        raise ValueError(f"Unsupported policy type: {type(args.policy)}")
    
    return UnifoLMOpenPIPolicy(
        ckpt_path=ckpt_path,
        vlm_pretrained_path=args.vlm_pretrained_path,
        unnorm_key=args.unnorm_key,
        center_crop=args.center_crop,
        use_bf16=args.use_bf16,
    )


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata
    
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating UnifoLM-VLA server (host: %s, ip: %s)", hostname, local_ip)
    
    if HAS_OPENPI:
        server = websocket_policy_server.WebsocketPolicyServer(
            policy=policy,
            host="0.0.0.0",
            port=args.port,
            metadata=policy_metadata,
        )
        logging.info(f"Starting OpenPI WebSocket server on ws://0.0.0.0:{args.port}")
        server.serve_forever()
    else:
        # Fallback to FastAPI if openpi not available
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
        import uvicorn
        
        app = FastAPI(title="UnifoLM-VLA Server", version="1.0")
        
        @app.post("/act")
        async def act(obs: dict):
            return JSONResponse(policy.infer(obs))
        
        logging.info(f"Starting FastAPI server on http://0.0.0.0:{args.port}")
        uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
