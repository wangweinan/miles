from pathlib import Path
from typing import Annotated

import typer
from pydantic import JsonValue

from miles.utils.file_utils import atomic_write_text
from miles.utils.test_utils.snapshot_comparison_models import (
    SnapshotComparisonRules,
    SnapshotFieldMove,
    SnapshotListSort,
)
from miles.utils.test_utils.snapshot_expansion import expand_snapshot_directory

app = typer.Typer(add_completion=False)

_TRAINER_BACKEND_FIELDS = (
    "rollout_indexer_topk_num_streams",
    "save",
    "save_interval",
    "async_save",
    "seed",
    "num_layers",
    "low_memory_resume",
    "distributed_backend",
    "distributed_timeout_minutes",
    "eval_interval",
    "global_batch_size",
    "micro_batch_size",
    "wandb_project",
)
_MEGATRON_BACKEND_FIELDS = (
    "lr",
    "bf16",
    "rotary_base",
    "mtp_detach_heads",
    "hidden_size",
    "num_attention_heads",
    "tensor_model_parallel_size",
    "pipeline_model_parallel_size",
    "context_parallel_size",
    "sequence_parallel",
)
_CHECKPOINT_FIELDS = ("load", "no_load_optim", "no_load_rng", "finetune", "ckpt_step")


@app.command()
def generate(
    before: Annotated[Path, typer.Option()],
    output: Annotated[Path, typer.Option()],
) -> None:
    assert not output.exists(), f"Rules already exist: {output}"
    snapshots = expand_snapshot_directory(directory=before)
    moves: list[SnapshotFieldMove] = []
    for case_name, case in snapshots.items():
        if case_name.startswith("parsed_config/"):
            assert isinstance(case, dict)
            if "config" in case:
                assert isinstance(config := case["config"], dict)
                moves.extend(_parsed_moves(args=config, scope=[case_name, "config"]))
            continue
        if not case_name.startswith("runtime_config/"):
            continue
        assert isinstance(case, dict)
        for process_name, process in case.items():
            assert isinstance(process, dict)
            if not process["ranks"]:
                continue
            stages = process["stages"]
            assert isinstance(stages, dict)
            for stage_name, stage in stages.items():
                assert isinstance(stage, dict) and isinstance(args := stage["args"], dict)
                scope = [case_name, process_name, "stages", stage_name, "args"]
                moves.extend(_moves(args=args, scope=scope, fields=_TRAINER_BACKEND_FIELDS, op="op30-47"))
                if args["train_backend"] == "megatron":
                    moves.extend(_moves(args=args, scope=scope, fields=_MEGATRON_BACKEND_FIELDS, op="op30-47"))
                if stage_name.startswith("checkpoint_load-") and args["train_backend"] == "megatron":
                    moves.extend(_moves(args=args, scope=scope, fields=_CHECKPOINT_FIELDS, op="op48-130/131"))
    list_sorts = [
        SnapshotListSort(
            path=["parsed_config/megatron_lora_targets.yaml", "config", name],
            op="op48-15",
            human_decision="2026-10-18: explicitly sort the two LoRA target lists for comparison",
        )
        for name in ("hf_lora_targets", "lora_adapter_targets")
        if "parsed_config/megatron_lora_targets.yaml" in snapshots
    ]
    rules = SnapshotComparisonRules(moves=moves, approved_differences=[], list_sorts=list_sorts)
    atomic_write_text(path=output, text=rules.model_dump_json(indent=2))


def _parsed_moves(*, args: dict[str, JsonValue], scope: list[str]) -> list[SnapshotFieldMove]:
    moves: list[SnapshotFieldMove] = []
    if args["train_backend"] == "megatron":
        moves.extend(
            _moves(
                args=args,
                scope=scope,
                target=("raw_megatron", "base_args"),
                op="op30-30",
                fields=_MEGATRON_BACKEND_FIELDS,
            )
        )
    elif args["train_backend"] == "fsdp":
        moves.extend(
            _moves(
                args=args,
                scope=scope,
                target=("raw_fsdp",),
                op="op30-64/op48-189",
                fields=(
                    "adam_beta1",
                    "adam_beta2",
                    "adam_eps",
                    "attn_implementation",
                    "bf16",
                    "clip_grad",
                    "dp_replicate_size",
                    "fsdp_cpu_backend",
                    "fsdp_cpu_offload",
                    "gradient_checkpointing",
                    "lr",
                    "mtp_detach_heads",
                    "mtp_loss_scaling_factor",
                    "mtp_num_layers",
                    "optimizer",
                    "warmup_ratio",
                    "weight_decay",
                    "kernel_backend",
                ),
            )
        )
    moves.extend(
        _moves(
            args=args,
            scope=scope,
            target=("sglang", "base_args"),
            source_prefix="sglang_",
            op="op30-6",
            fields=("tp_size", "dp_size", "dtype", "enable_metrics", "mem_fraction_static", "speculative_algorithm"),
        )
    )
    moves.extend(
        _moves(
            args=args,
            scope=scope,
            target=("router_args",),
            source_prefix="router_",
            op="op30-28",
            fields=("policy", "request_timeout_secs", "health_check_timeout_secs", "max_concurrent_requests"),
        )
    )
    return moves


def _moves(
    *,
    args: dict[str, JsonValue],
    scope: list[str],
    fields: tuple[str, ...],
    op: str,
    target: tuple[str, ...] = ("backend",),
    source_prefix: str = "",
) -> list[SnapshotFieldMove]:
    return [
        SnapshotFieldMove(
            source=[*scope, source_prefix + name],
            target=[*scope, *target, name],
            op=op,
            reason="Move the explicitly listed input to its configuration owner without changing its value",
        )
        for name in fields
        if source_prefix + name in args
    ]


if __name__ == "__main__":
    app()
