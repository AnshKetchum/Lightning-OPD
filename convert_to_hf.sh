MEGATRON_CKPT_DIR=/workspace/Lightning-OPD/models/Qwen3-4B-Base-Open-Thoughts-Qwen3-8B-sft-3k_ckpt__qwen3-4b-lightning-opd/iter_0001009 \
HF_OUTPUT_DIR=/workspace/Lightning-OPD/models/Qwen3-4B-Base-Open-Thoughts-Qwen3-8B-sft-3k_hf__qwen3-4b-lightning-opd \
ORIGIN_HF_DIR=/workspace/Lightning-OPD/checkpoints/qwen3-4b-base-sft-qwen3-8b/step_3000 \
bash scripts/convert_megatron_to_hf.sh