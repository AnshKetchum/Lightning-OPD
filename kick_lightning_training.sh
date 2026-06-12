export SFT_CHECKPOINT=checkpoints/qwen3-4b-base-sft-qwen3-8b/step_3000
export LIGHTNING_OPD_DATA=data/lightning_opd/dapo-math-17k-qwen3-4b-sft-rollouts-lightning-opd-precomputed.parquet

python configs/lightning_opd/qwen3-4b-lightning-opd.py