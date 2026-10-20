# miles on TPU

GRPO training of Qwen3-4B on one TPU v7x 2x2x1 host (2 chips per half, 2 TensorCores per chip):
the torchtitan trainer runs as 4 ranks on devices 0-3 (chips 0-1), and sglang-torchtpu serves
rollouts on devices 4-7 (chips 2-3) as an external engine. Weights move trainer → server
through shared memory (`--update-weight-transfer-mode shm`, about 3 s per sync for Qwen3-4B;
`disk` takes about 5 s).

## Environment

The two halves need different PyTorch stacks, so they run from different Python environments.

**sglang** runs from the sglang-torchtpu image's system Python (torch 2.11, torch_tpu 0.1.1).
Nothing to install: its launcher only needs this checkout on `PYTHONPATH`.

**The trainer** needs torch 2.13+ for torchtitan. Build a venv next to the image's Python:

```bash
python3 -m venv /path/to/venvs/train
source /path/to/venvs/train/bin/activate

# torch 2.13 (CPU wheel; torch_tpu provides the device)
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu

# torch_tpu nightly (ships glue for torch 2.11-2.15); needs gcloud credentials
pip install torch_tpu --pre \
  --extra-index-url "https://oauth2accesstoken:$(gcloud auth print-access-token)@us-python.pkg.dev/ml-oss-artifacts-transient/torch-tpu-virtual-registry/simple/"

# torchtitan v0.3.0 (the commit miles pins) with miles' torch_tpu patch: device calls, FSDP reduce, compile
git clone https://github.com/pytorch/torchtitan.git && cd torchtitan
git checkout $(sed -n 's/^ARG TORCHTITAN_COMMIT=//p' /path/to/miles/docker/Dockerfile)
patch --fuzz=0 -p1 < /path/to/miles/docker/patch/torchtitan_torch_tpu.patch
pip install -e . && cd ..

# miles itself
pip install -e /path/to/miles
```

## Run

Model and data default to `/root/models` and `/root/datasets` (`--model-dir`, `--data-dir`);
`train` downloads Qwen3-4B, dapo-math-17k and aime-2024 there if missing.

1. Start sglang on devices 4-7 from the image's Python, in its own shell, and wait for
   `curl localhost:30000/health` to return 200 (about 5 minutes, mostly compilation):

   ```bash
   PYTHONPATH=$PWD python3 scripts/tpu/run_qwen3_4b_torchtitan.py sglang
   ```

2. Train on devices 0-3 from the trainer venv:

   ```bash
   source /path/to/venvs/train/bin/activate
   python3 scripts/tpu/run_qwen3_4b_torchtitan.py train --recipe full
   ```

   `--recipe smoke` runs 3 short rollouts to check the setup; `--recipe bench` runs 4 rollouts
   at full shapes without eval, for timing. Pass extra `train.py` flags through `--extra-args`.

wandb is on when `WANDB_API_KEY` is set or `~/.netrc` holds the key; `--wandb-project ""`
turns it off.

## Performance

Qwen3-4B, 128 samples × up to 7168 response tokens per step, `--recipe bench`:

| | ref log-probs | log-probs | actor train | train step |
|---|---|---|---|---|
| eager FSDP | 24.5 s | 29.0 s | 83.8 s | 145.6 s |
| `--enable-compile` (default) | 10-12 s | 13 s | 32-34 s | 62-67 s |

The compiled path adds per-block `torch.compile` on the `tpu` backend, FSDP gather-once
(`--titan-fsdp-gather-once`), and packed-document attention on native SDPA
(`miles/backends/torchtitan_utils/tpu_doc_attention.py`, installed automatically on TPU).

## fp32 SiLU

v7x evaluates a bf16 SiLU in bf16, which makes Qwen3's logits noisier than on v6e.
`--activation-in-fp32` (pass it to both `sglang` and `train`) computes `silu(gate)` in fp32 in the
trainer (`train.py --activation-in-fp32`) and in the engine (`SGLANG_TPU_ACTIVATION_FP32=1`, needs an
sglang with that switch); matmuls stay bf16. Turn it on for both sides or neither: fixing one side
only widens the train-vs-rollout gap, so `train` refuses an engine whose setting differs.

## Notes

- **Stop a training run gracefully while sglang is running.** The host's TPU network session is
  shared, and a `kill -9`'d TPU process can take the other half down with it. Stop a run with
  `ray stop` (without `--force`) and wait for its devices to be released before starting the next:
  `train` force-stops any ray processes still left from an earlier run (it leaves sglang alone).
- **Do not mask chips with `TPU_VISIBLE_CHIPS`**: it hangs libtpu's session setup. Both halves
  address their devices through `TORCH_TPU_TOPOLOGY=1,2,1,2` instead; the launcher sets it.
- **Keep crash dumps off small disks** (`ulimit -c 0`): core files from crashed TPU processes
  can be tens of GB.
