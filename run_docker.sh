# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

docker run -it --runtime=nvidia \
    --shm-size=64g \
    -v $(pwd):/workspace/Lightning-OPD \
    -v $HOME/.cache:$HOME/.cache \
    -v /scratch/ansh/:/scratch/ansh \
    -w /workspace/Lightning-OPD \
    tonyhao96/jetmoe:v0.2 \
    bash
