# DegroidVideo: A Systematic Framework For Large Video Generation Model

This repo contains PyTorch model definitions, pre-trained weights and inference/sampling code for DegroidVideo.

> [**DegroidVideo: A Systematic Framework For Large Video Generation Model**](https://arxiv.org/abs/2412.03603)

## Abstract

We present DegroidVideo, a novel open-source video foundation model that exhibits performance in video generation comparable to leading closed-source models. We adopt key technologies including data curation, image-video joint model training, and efficient infrastructure for large-scale training and inference. Through effective scaling of model architecture and dataset, we successfully trained a video generative model with over 13 billion parameters.

According to professional human evaluation results, DegroidVideo outperforms previous state-of-the-art models. By releasing the code and weights, we aim to bridge the gap between closed-source and open-source video foundation models.

## DegroidVideo Overall Architecture

DegroidVideo is trained on a spatial-temporally compressed latent space via a Causal 3D VAE. Text prompts are encoded using a large language model. Taking Gaussian noise and conditions as input, the generative model produces an output latent, which is decoded through the 3D VAE decoder.

## Key Features

### Unified Image and Video Generative Architecture

DegroidVideo introduces Transformer design with Full Attention mechanism for unified image and video generation. A "Dual-stream to Single-stream" hybrid model design is used. In the dual-stream phase, video and text tokens are processed independently. In the single-stream phase, tokens are concatenated for effective multimodal information fusion.

### MLLM Text Encoder

We utilize a pre-trained Multimodal Large Language Model (MLLM) with a Decoder-Only structure as our text encoder, offering better image-text alignment, superior detail description and complex reasoning, and zero-shot learning capabilities.

### 3D VAE

DegroidVideo trains a 3D VAE with CausalConv3D to compress pixel-space videos into a compact latent space. Compression ratios: video length (4x), space (8x), channel (16x).

### Prompt Rewrite

We fine-tune a model as our prompt rewrite system to adapt user prompts to model-preferred prompts. Two modes are available: Normal mode (enhances comprehension) and Master mode (enhances visual quality aspects).

## Requirements

The following requirements apply for running DegroidVideo (batch size = 1):

| Model | Setting (height/width/frame) | GPU Peak Memory |
|-------|------------------------------|-----------------|
| DegroidVideo | 720px1280px129f | 60GB |
| DegroidVideo | 544px960px129f | 45GB |

- NVIDIA GPU with CUDA support required
- Minimum: 60GB for 720p, 45GB for 544p
- Recommended: 80GB GPU
- Tested OS: Linux

## Dependencies and Installation

```shell
git clone https://github.com/Tencent-Degroid/DegroidVideo
cd DegroidVideo

# Create conda environment
conda create -n DegroidVideo python==3.10.9
conda activate DegroidVideo

# Install PyTorch (CUDA 12.4)
conda install pytorch==2.6.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia

# Install dependencies
python -m pip install -r requirements.txt

# Install flash attention v2
python -m pip install ninja
python -m pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.6.3

# Install xDiT for parallel inference
python -m pip install xfuser==0.4.0
