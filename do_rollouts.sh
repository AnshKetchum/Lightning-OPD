SFT_CHECKPOINT=checkpoints/qwen3-4b-base-sft-qwen3-8b/step_3000 \
OPD_PROMPTS=data/prompts/dapo-math-17k/dapo-math-17k.jsonl \
OUTPUT_DIR=data/rollouts \
bash scripts/collect_rollouts.sh 2>&1 | tee rollout_out.log