import torch
import torch.ao.quantization
from torch.ao.quantization import get_default_qconfig, prepare, convert
import torch.nn as nn
from torchvision.models import vision_transformer

import timm

import utils.eval_utils as eval_utils


def return_split_qkv_vit_b16():
    def remove_stubs_recursive(module):
        for name, child in list(module.named_children()):
            if isinstance(
                child,
                (
                    torch.ao.quantization.QuantStub,
                    torch.ao.quantization.DeQuantStub,
                    torch.ao.quantization.HistogramObserver,
                ),
            ):
                delattr(module, name)
            if "q_scaling" in name:
                delattr(module, name)
            else:
                remove_stubs_recursive(child)

    model = vision_transformer.vit_b_16(pretrained=True)
    print(model)
    model.eval()

    print("Evaluating torch vanilla ViT...")
    config_path = "config/vit_s16.yaml"
    config = eval_utils.parse_config(config_path)
    train_loader, val_loader = eval_utils.load_data(**config.data)
    with torch.no_grad():
        acc1, acc5 = eval_utils.validate_model(val_loader, model)
    print(f"Top-1 accuracy: {acc1:.2f}, Top-5 accuracy: {acc5:.2f}")

    model.qconfig = get_default_qconfig("fbgemm")
    prepare(model, inplace=True)

    # Remove all QuantStub and DeQuantStub recursively
    model.apply(torch.ao.quantization.disable_observer)
    model.apply(torch.ao.quantization.disable_fake_quant)

    print(model)

    model.eval()
    model.cuda()
    print("Evaluating split QKV ViT...")
    config_path = "config/vit_s16.yaml"
    config = eval_utils.parse_config(config_path)
    train_loader, val_loader = eval_utils.load_data(**config.data)
    with torch.no_grad():
        acc1, acc5 = eval_utils.validate_model(val_loader, model)
    print(f"Top-1 accuracy: {acc1:.2f}, Top-5 accuracy: {acc5:.2f}")


import torch
import torch.nn as nn
import timm
from timm.models.vision_transformer import Attention
import gc
from types import MethodType

from run_ptmq import (
    quantize_model,
    search_fold_and_remove_bn,
    disable_all,
    set_qmodel_block_aqbit,
)
from quant.quant_module import QuantizedLayer, QuantizedBlock, Quantizer


class AttentionSplitQKV(Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        dim = self.num_heads * self.head_dim
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)

        print(f"self.qkv.weight.shape: {self.qkv.weight.shape}")
        v, q, k = self.qkv.weight.data.chunk(3, dim=0)
        self.q_proj.weight.data.copy_(q)
        self.k_proj.weight.data.copy_(k)
        self.v_proj.weight.data.copy_(v)

        # Remove the original qkv projection to prevent redundancy
        del self.qkv
        gc.collect()

    def forward(self, x):
        B, N, C = x.shape

        # FIND Q, K, V, X SHAPES
        # REPLICATE QKV OUTPUT!

        q = (
            self.q_proj(x)
            .reshape(B, N, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k_proj(x)
            .reshape(B, N, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v_proj(x)
            .reshape(B, N, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )

        print(f"q, k, v shapes: {q.shape}, {k.shape}, {v.shape}")

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        del q, k
        gc.collect()

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        del attn, v
        gc.collect()
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


def split_timm_qkv_vit():
    model = timm.create_model("vit_small_patch16_224", pretrained=True)
    config_path = "config/vit_s16.yaml"

    config = eval_utils.parse_config(config_path)
    train_loader, val_loader = eval_utils.load_data(**config.data)

    # Replace all Attention modules with AttentionSplitQKV
    def replace_attention(module):
        for name, child in module.named_children():
            if isinstance(child, Attention):
                dim = child.num_heads * child.head_dim
                setattr(
                    module,
                    name,
                    AttentionSplitQKV(
                        dim,
                        child.num_heads,
                        qkv_bias=False,
                        attn_drop=0.0,
                        proj_drop=0.0,
                    ),
                )
            else:
                replace_attention(child)

    replace_attention(model)

    search_fold_and_remove_bn(model)
    model = quantize_model(model, config)

    # print quantized model
    print(model)

    # disable_all(model)
    # for name, module in model.named_modules():
    #     if isinstance(module, QuantizedBlock):
    #         module.qoutput = False

    model.eval()
    model.cuda()
    print("Evaluating split QKV ViT...")
    with torch.no_grad():
        acc1, acc5 = eval_utils.validate_model(val_loader, model)
    print(f"Top-1 accuracy: {acc1:.2f}, Top-5 accuracy: {acc5:.2f}")

    # for name, module in model.named_modules():
    #     print(name, type(module), end="")
    #     if hasattr(module, 'weight'):
    #         print(module.weight.shape)
    #     else:
    #         print()


def get_torch_summary():
    import torchsummary

    model = timm.create_model("regnetx_006", pretrained=True)
    model.cuda()

    # torchsummary.summary(model, input_size=(3, 224, 224), batch_size=1)

    for name, module in model.named_modules():
        print(name, type(module), end="")
        if hasattr(module, "weight"):
            print(module.weight.shape)
        else:
            print()


if __name__ == "__main__":
    # split_timm_qkv_vit()
    get_torch_summary()
