from dataclasses import dataclass

import torch

from miles.backends.fsdp_utils.arguments import FSDPArgs, build_dataclass_parser, load_args_from_parser
from miles.utils import accelerator

_TPU_ATTENTION_BACKENDS = ("sdpa_blocks", "splash")


@dataclass
class TorchtitanArgs(FSDPArgs):
    # torchtitan.models.<name>.model_registry(<flavor>)
    titan_model_name: str = "qwen3"
    titan_model_flavor: str = "0.6B"

    seq_length: int = 4096

    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    expert_model_parallel_size: int = 1

    titan_compile: bool = False  # torch.compile each TransformerBlock
    titan_fsdp_gather_once: bool = False  # all-gather FSDP params once per pass, not per microbatch
    # packed-document attention on TPU: "sdpa_blocks" (query blocks over SDPA) or "splash" (Pallas kernel)
    titan_tpu_attention: str = "sdpa_blocks"


def build_torchtitan_parser(extra_args_provider=None):
    return build_dataclass_parser(TorchtitanArgs, "torchtitan Training (miles)", extra_args_provider)


def load_torchtitan_args(extra_args_provider=None):
    return load_args_from_parser(build_torchtitan_parser(extra_args_provider))


def validate_torchtitan_args(args) -> None:
    torch_version = tuple(int(part) for part in torch.__version__.split(".")[:2])
    if torch_version < (2, 13):
        raise ValueError(f"the torchtitan backend needs torch>=2.13; this environment runs {torch.__version__}")
    if args.context_parallel_size != 1 and args.titan_model_name == "qwen3_5":
        raise ValueError("torchtitan does not support context parallelism for qwen3_5")

    if args.rollout_max_context_len is not None:
        if args.seq_length < args.rollout_max_context_len:
            raise ValueError(
                f"--seq-length {args.seq_length} is shorter than "
                f"--rollout-max-context-len {args.rollout_max_context_len}: torchtitan builds its "
                "rotary embeddings for the former, so a longer sequence would index past them"
            )
    elif args.seq_length <= args.rollout_max_response_len:
        raise ValueError(
            f"--seq-length {args.seq_length} leaves no room for a prompt ahead of "
            f"--rollout-max-response-len {args.rollout_max_response_len}: torchtitan builds its "
            "rotary embeddings for the former, and a prompt-plus-response beyond them asserts "
            "inside the rope kernel"
        )

    if args.titan_tpu_attention not in _TPU_ATTENTION_BACKENDS:
        raise ValueError(
            f"--titan-tpu-attention {args.titan_tpu_attention!r}: expected one of {_TPU_ATTENTION_BACKENDS}"
        )
    if args.titan_tpu_attention != "sdpa_blocks" and not accelerator.IS_TPU:
        raise ValueError(f"--titan-tpu-attention {args.titan_tpu_attention} applies on TPU only")
    if args.titan_tpu_attention == "splash" and args.seq_length % 1024:
        raise ValueError(
            f"--titan-tpu-attention splash tiles the sequence in 1024-token blocks; --seq-length {args.seq_length} "
            "is not a multiple of 1024"
        )
    if accelerator.IS_TPU and args.log_probs_backend == "fused":
        raise ValueError("--log-probs-backend fused is a Triton kernel; on TPU use --log-probs-backend torch")
    if accelerator.IS_TPU and args.use_dynamic_batch_size and args.max_tokens_per_gpu > args.seq_length:
        raise ValueError(
            f"--max-tokens-per-gpu {args.max_tokens_per_gpu} exceeds --seq-length {args.seq_length}: on TPU "
            "packed microbatches are padded to a power-of-two bucket capped at --seq-length"
        )

    if args.titan_fsdp_gather_once and args.pipeline_model_parallel_size > 1:
        raise ValueError("--titan-fsdp-gather-once does not apply to the pipeline schedule; drop it under PP")
    if args.ref_update_interval is not None:
        raise ValueError("--ref-update-interval is not supported by the torchtitan backend")
    if args.fp16:
        raise ValueError("the torchtitan backend trains in bf16 mixed precision; --fp16 is not supported")
    if args.lr_decay_style not in ("constant", "linear", "cosine"):
        raise ValueError(
            f"torchtitan's LR schedule has no {args.lr_decay_style!r} decay; use constant, linear or cosine"
        )
    if args.lr_warmup_fraction is not None or args.lr_wsd_decay_iters is not None or args.lr_decay_iters is not None:
        raise ValueError(
            "torchtitan's LR schedule takes --lr-warmup-iters only; fraction, WSD and decay-iters are unsupported"
        )
    if args.save_debug_train_data is not None:
        raise ValueError("--save-debug-train-data is not wired up for the torchtitan backend")
