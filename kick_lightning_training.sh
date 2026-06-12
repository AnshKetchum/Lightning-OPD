export SFT_CHECKPOINT=checkpoints/qwen3-4b-base-sft-qwen3-8b/step_3000
export LIGHTNING_OPD_DATA=data/lightning_opd/dapo-math-17k-qwen3-4b-sft-rollouts-lightning-opd-precomputed-student-logprobs.parquet

export WANDB_KEY=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
# python configs/lightning_opd/qwen3-4b-lightning-opd.py
PYTHONPATH=/workspace/Lightning-OPD python configs/lightning_opd/qwen3-4b-lightning-opd.py 2>&1 | tee out.log