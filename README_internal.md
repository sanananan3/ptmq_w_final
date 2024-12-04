****<center>
<img src="./imgs/ptmq_fig1.png" width="500px"></img>
</center>

# ptmq-pytorch

PyTorch implementation of [PTMQ: Post-training Multi-Bit Quantization of Neural Networks (Xu et al., AAAI 2024)]((https://ojs.aaai.org/index.php/AAAI/article/view/29553)).

---

## Getting Started

### Running PTMQ

```bash
python run_ptmq.py --config configs/[config_file].yaml
```

Create your own configuration file in the `configs` directory.

### Compatible models

We aim to replicate all baselines from the paper:

- ✅All ResNet models (18, 34, 50, 101, 152)
- ✅MobileNetV2
- ✅ViT (ViT-S/224/16, ViT-B/224/16)
- ✅DeiT (DeiT-S/224/16, DeiT-B/224/16)
- ✅RegNetX-600

---

## Useful Commands

### Initial Setup for Cloud GPUs ([runpod.io](https://runpod.io?ref=9t3u4v13))

```bash
# create virtual environment and install dependencies
python -m venv z_venv
source z_venv/bin/activate
pip install --upgrade pip
pip install torch torchvision easydict PyYAML scipy gdown timm nvitop kaggle wandb torchsummary
pip install git+https://github.com/psf/black

# download resnet18 weights
cd ~/dev/ptmq-pytorch
python
import torch
import torchvision.models as models
resnet18 = models.resnet18(pretrained=True)
torch.save(resnet18.state_dict(), "resnet18_weights.pth")
exit()

# download resnet50 weights
cd ~/dev/ptmq-pytorch
python
import torch
import torchvision.models as models
resnet50 = models.resnet50(pretrained=True)
torch.save(resnet50.state_dict(), "resnet50_weights.pth")
exit()

# download mobilenetv2 weights
cd ~/dev/ptmq-pytorch
python
import torch
import torchvision.models as models
mobilenetv2 = models.mobilenet_v2(pretrained=True)
torch.save(mobilenetv2.state_dict(), "mobilenetv2_weights.pth")
exit()

# download vit weights - pytorch ViT is too buggy, use `timm`
cd ~/dev/ptmq-pytorch
python
import torch
from timm.models import create_model
model = create_model(
    'vit_small_patch16_224.augreg_in1k',  # Model configuration
    pretrained=True,  # Load pre-trained weights
    img_size=224      # Image size, adjust if needed
)
torch.save(model.state_dict(), 'vit_s16_pretrained.pth')
exit()
```


```bash
# login for wandb
pip install wandb
wandb login
# enter wandb API key from https://wandb.ai/authorize
```

### Downloading Datasets

Use `imagenet-mini/train` for calibration, and `imagenet/val` (the full ImageNet1K validation set) for evaluation.

#### Mini-ImageNet from Kaggle

```bash
pip install kaggle
cd ~/dev
mkdir -p ~/dev/kaggle # add kaggle.json with {"username":"xxx","key":"xxx"} here
chmod 600 ~/dev/kaggle/kaggle.json
kaggle datasets download -d ifigotin/imagenetmini-1000
apt-get update && apt-get install unzip
unzip ~/dev/imagenetmini-1000.zip -d ~/dev
```

#### ImageNet Validation Dataset

```bash
# download imagenet validation dataset (from public google drive)
mkdir -p ~/dev/imagenet/val
cd imagenet/val
gdown https://drive.google.com/uc?id=11omFedOvjslBRMFc-lrM3n2t0xP99FXB -O ILSVRC2012_img_val.tar
tar -xvf ILSVRC2012_img_val.tar
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
```

### `wandb` setup

```bash
pip install wandb
wandb login # enter API key
```

---

## Overview

PTMQ (post-training multi-bit quantization) is a post-training quantization method that performs **block-level activation quantization** with multiple bit-widths.

In a sense, it can be viewed as a knowledge distillation method, where the higher-precision features are used to guide the quantization of lower-precision features.

For starters, we can think of the target inference model to be W3A3. PTMQ has separate strategies in order to ensure that the weights and activations are better quantized. These are all employed during the block reconstruction phase, and weights and activations are optimized with a reconstruction loss (GD loss) and round loss, respectively.

**Weights** are better quantized using rounding optimization via AdaRound (Nagel et al., ICML 2020). This is done by minimizing the quantization error of the weights.

**Activations** are better quantized by using a multi-bit feature mixer, which is 3 separate quantizers for low, medium, and high bit-widths. We learn activation step sizes (Esser et al., ICLR 2020) to minimize the activation quantization error, via a group-wise distillation loss.

The novelty of this model is that through block reconstruction, we can quickly and efficiently quantize a full-precision model to multiple bit-widths, which can be flexibly be deployed based on the given hardware constraints in real-time.

---

## Reproducing Results

### ResNet-18

- iterations: 5000
- block reconstruction hyperparameters:
  - batch_size: 32
  - scale_lr: 4.0e-5
  - warm_up: 0.2
  - weight: 0.01
  - iters: 5000 #20000
  - b_range: [20, 2]
  - keep_gpu: True
  - round_mode: learned_hard_sigmoid
  - mixed_p: 0.5
- ptmq hyperparameters:
  - lambda1: 0.4
  - lambda2: 0.3
  - lambda3: 0.3
  - mixed_p: 0.5
  - gamma1: 100
  - gamma2: 100
  - gamma3: 100
- first and last layer weights: 8-bit
- last layer activations: 8-bit
- low, medium, high bit-widths for each precision (not mentioned in paper)
  - W3A3: (l, m, h) = (3, 4, 5)
  - W4A4: (l, m, h) = (4, 5, 6)
  - W5A5: (l, m, h) = (5, 6, 7)
  - W6A6: (l, m, h) = (6, 7, 8)

`ResNet-18`
| Info | Precision Type | GPU | Time (min) | W3A3 | W4A4 | W5A5 | W6A6 | **W32A32** |
| ----- | ------------- | --- | ---------- | ---- | ---- | ---- | ---- | ------ |
| Paper | Mixed | Nvidia 3090 | 100 | 64.02 | 67.57 | 69.00 | 70.23 | **71.08** |
| Our Code | Uniform | Nvidia A40 | 14.41 | 63.47 | 67.59 | 69.02 | 69.50 | **71.08** |

We compare how relative bit-precision affects our desired performance precision.

| Precision Type | Precision | Mixed-bit (l,m,h) | Top-1 (%) |
| -------------- | --------- | ----------------- | --------- |
| - | W32A32 | - | 71.08 |
| Mixed | W5A5 | **Baseline** (unknown) | 69.00 |
| Uniform | W5A5 | (3, 4, **5**) | 65.95 |
| Uniform | W5A5 | (4, **5**, 6) | 67.90 |
| Uniform | W5A5 | (**5**, 6, 7) | 69.02 |



### Vision Transformer

- when using ViT from `timm`, we have to use proper transforms (see [here](https://huggingface.co/timm/vit_small_patch16_224.augreg_in1k#image-classification)) - otherwise performance is significantly worse
- see [here](https://github.com/huggingface/pytorch-image-models/blob/main/results/results-imagenet.csv) for performance of `timm` models on ImageNet
- we empirically find that block reconstruction for vision transformers with `MSEObserver` takes a while
- using `MinMaxObserver` with an aggressive learning rate (e.g. 4e-3, 4e-2) can enable faster convergence

| Source | Model | W4A6 | W5A6 | W6A6 | W7A7 | W8A8 | FP32 |
| ------ | ----- | ---- | ---- | ---- | ---- | ---- | ---- |
| PTMQ (Paper) | ViT-S/16 | 71.67 | 75.14 | 76.09 | 77.14 | 78.16 | 78.84 |
| PTMQ (Our Code) | ViT-S/16 | - | - | - | - | - | 78.84 |

#### `iter`=1000, `lr`=4e-2, 
- Paper: W4A6 = 71.67
  - (l,m,h) = (4, 5, **6**): 1.21, 1.39, **1.45**
  - (l,m,h) = (2, **6**, 8): 0.10, **0.09**, 0.12
  - (l,m,h) = (**6**, 7, 8): **74.33**, 75.84, 76.22
- Paper: W5A6 = 75.14
  - (l,m,h) = (4, 5, **6**): 0.61, 0.67, **0.65**
  - (l,m,h) = (2, **6**, 8): 0.10, **0.07**, 0.09
  - (l,m,h) = (**6**, 7, 8): **73.27**, 75.63, 75.82
- Paper: W6A6 = 76.09
  - (l,m,h) = (4, 5, **6**):
  - (l,m,h) = (2, **6**, 8):
  - (l,m,h) = (**6**, 7, 8): **69.38**, 72.69, 72.44 
- Paper: W7A7 = 77.14
  - (l,m,h) = (5, 6, **7**):
  - (l,m,h) = (2, **7**, 8):
  - (l,m,h) = (6, **7**, 8): 66.87, **69.74**, 69.46
- Paper: W8A8 = 78.16
  - (l,m,h) = (6, 7, **8**): 66.87, 66.63, **66.51**


| Source | Model | W4A6 | W5A6 | W6A6 | W7A7 | W8A8 | FP32 |
| ------ | ----- | ---- | ---- | ---- | ---- | ---- | ---- |
| PTQ4ViT (Paper) | ViT-B/16 | 75.43 | 76.17 | 77.66 | 78.85 | 78.98 | 81.07 |
| PTMQ (Paper) | ViT-B/16 | 75.00 | 76.64 | 77.70 | 78.62 | 79.12 | 81.07 |
| PTMQ (Our Code) | ViT-B/16 | 74.33 | 73.27 | 69.38 | 69.74 | 66.51 | 81.07 |


#### W6A6, `iter` and `lr` experiments

| Source | Model | `lr` | `iter` | W6A6 |
| ------ | ----- | ---- | ------ | ---- |
| PTMQ (Paper) | ViT-B/16 | - | - | 77.70 |
| PTMQ (Our Code) | ViT-B/16 | 4e-2 | 1000 | 69.38 |
| PTMQ (Our Code) | ViT-B/16 | 4e-3 | 1000 | 63.39 |
| PTMQ (Our Code) | ViT-B/16 | 4e-3 | 3000 | 74.75 |
| PTMQ (Our Code) | ViT-B/16 | 4e-3 | 5000 | 75.89 |

---


## References

```bibtex
@article{Xu_Li_Wang_Zhang_2024,
  title={PTMQ: Post-training Multi-Bit Quantization of Neural Networks},
  volume={38},
  url={https://ojs.aaai.org/index.php/AAAI/article/view/29553},
  DOI={10.1609/aaai.v38i14.29553},
  number={14},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence},
  author={Xu, Ke and Li, Zhongcheng and Wang, Shanshan and Zhang, Xingyi},
  year={2024},
  month={Mar.},
  pages={16193-16201}
}
```

- source code for [QDrop: Randomly Dropping Quantization for Extremely Low-bit Post-Training Quantization](https://github.com/wimh966/QDrop) was used as a main reference

## TODO

- [x] PTMQ - Key Contributions
  - [x] Multi-bit Feature Mixer (MFM)
  - [x] Group-wise Distill Loss (GD-Loss)
- [x] Fundamental Tools
  - [x] Rounding-based quantization (AdaRound)
  - [x] BatchNorm folding
- [x] Quantization Modules
  - [x] Layer quantization
  - [x] Block quantization
  - [x] Model quantization
- [x] Reconstruction
  - [x] Block Reconstruction
- [x] PTMQ - Sanity Test
  - [x] CNN - ResNet-18
  - [x] Transformer - ViT
- [ ] Preliminary Results
  - [ ] PTMQ verification
    - [x] CNN - ResNet-18
    - [ ] 🔥Transformer - ViT
- [ ] Final Overview
  - [ ] verify on most/all experiments
  - [ ] (partially) reproduce results