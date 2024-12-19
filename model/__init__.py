import torch.nn as nn
import torch
import torchvision
from utils import hubconf  # Changed from relative import to absolute import

from .resnet import BasicBlock, Bottleneck, resnet18, resnet50  # noqa: F401
from .regnet import ResBottleneckBlock, regnetx_600m, regnetx_3200m  # noqa: F401
from .mobilenetv2 import InvertedResidual, mobilenetv2  # noqa: F401
from .mnasnet import _InvertedResidual, mnasnet  # noqa: F401
from quant.quant_module import QuantizedLayer, QuantizedBlock, Quantizer   # noqa: F401

import timm
from timm.models.regnet import Bottleneck as RegNetBottleneck
from timm.models.vision_transformer import Block as VITBlock


class QuantBasicBlock(QuantizedBlock):
    """
    Implementation of Quantized BasicBlock used in ResNet-18 and ResNet-34.
    """
    def __init__(self, org_module: BasicBlock, config, qoutput=True, out_mode="calib"):
        super().__init__()
        self.out_mode = out_mode
        self.qoutput = qoutput
        
        self.conv1_relu = QuantizedLayer(org_module.conv1, org_module.relu1, config)
        self.conv2 = QuantizedLayer(org_module.conv2, None, config, qoutput=False)
        if org_module.downsample is None:
            self.downsample = None
        else:
            self.downsample = QuantizedLayer(org_module.downsample[0], None, config, qoutput=False)
        self.activation = org_module.relu2
        
        if self.qoutput:
            self.block_post_act_fake_quantize_low = Quantizer(None, config.quant.a_qconfig_low)
            self.block_post_act_fake_quantize_med = Quantizer(None, config.quant.a_qconfig_med)
            self.block_post_act_fake_quantize_high = Quantizer(None, config.quant.a_qconfig_high)
            
            self.f_l = None
            self.f_m = None
            self.f_h = None
            self.f_lmh = None
            
            self.lambda1 = config.quant.ptmq.lambda1
            self.lambda2 = config.quant.ptmq.lambda2
            self.lambda3 = config.quant.ptmq.lambda3
            
            self.mixed_p = config.quant.ptmq.mixed_p

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        x = self.conv1_relu(x)
        x = self.conv2(x)
        x += residual
        x = self.activation(x)
        
        if self.qoutput:
            if self.out_mode == "calib":
                self.f_l = self.block_post_act_fake_quantize_low(x)
                self.f_m = self.block_post_act_fake_quantize_med(x)
                self.f_h = self.block_post_act_fake_quantize_high(x)

                self.f_lmh = (
                    self.lambda1 * self.f_l
                    + self.lambda2 * self.f_m
                    + self.lambda3 * self.f_h
                )
                f_mixed = torch.where(torch.rand_like(x) < self.mixed_p, x, self.f_lmh)

                x = f_mixed
            elif self.out_mode == "low":
                x = self.block_post_act_fake_quantize_low(x)
            elif self.out_mode == "med":
                x = self.block_post_act_fake_quantize_med(x)
            elif self.out_mode == "high":
                x = self.block_post_act_fake_quantize_high(x)
            else:
                raise ValueError(
                    f"Invalid out_mode '{self.out_mode}': only ['low', 'med', 'high'] are supported"
                )
        return x


class QuantBottleneck(QuantizedBlock):
    """
    Implementation of Quantized Bottleneck Block used in ResNet-50, -101 and -152.
    """
    def __init__(self, org_module: Bottleneck, config, qoutput=True, out_mode="calib"):
        super().__init__()
        self.out_mode = out_mode
        self.qoutput = qoutput
        
        self.conv1_relu = QuantizedLayer(org_module.conv1, org_module.relu1, config)
        self.conv2_relu = QuantizedLayer(org_module.conv2, org_module.relu2, config)
        self.conv3 = QuantizedLayer(org_module.conv3, None, config, qoutput=False)

        if org_module.downsample is None:
            self.downsample = None
        else:
            self.downsample = QuantizedLayer(org_module.downsample[0], None, config, qoutput=False)
        self.activation = org_module.relu3
        
        if self.qoutput:
            self.block_post_act_fake_quantize_low = Quantizer(None, config.quant.a_qconfig_low)
            self.block_post_act_fake_quantize_med = Quantizer(None, config.quant.a_qconfig_med)
            self.block_post_act_fake_quantize_high = Quantizer(None, config.quant.a_qconfig_high)

            self.f_l = None
            self.f_m = None
            self.f_h = None
            self.f_lmh = None

            self.lambda1 = config.quant.ptmq.lambda1
            self.lambda2 = config.quant.ptmq.lambda2
            self.lambda3 = config.quant.ptmq.lambda3

            self.mixed_p = config.quant.ptmq.mixed_p

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        x = self.conv1_relu(x)
        x = self.conv2_relu(x)
        x = self.conv3(x)
        x += residual
        x = self.activation(x)
        
        if self.qoutput:
            if self.out_mode == "calib":
                self.f_l = self.block_post_act_fake_quantize_low(x)
                self.f_m = self.block_post_act_fake_quantize_med(x)
                self.f_h = self.block_post_act_fake_quantize_high(x)

                self.f_lmh = (
                    self.lambda1 * self.f_l
                    + self.lambda2 * self.f_m
                    + self.lambda3 * self.f_h
                )
                f_mixed = torch.where(torch.rand_like(x) < self.mixed_p, x, self.f_lmh)

                x = f_mixed
            elif self.out_mode == "low":
                x = self.block_post_act_fake_quantize_low(x)
            elif self.out_mode == "med":
                x = self.block_post_act_fake_quantize_med(x)
            elif self.out_mode == "high":
                x = self.block_post_act_fake_quantize_high(x)
            else:
                raise ValueError(
                    f"Invalid out_mode '{self.out_mode}': only ['low', 'med', 'high'] are supported"
                )
        return x

# 3. QuantRegNetBottleneck -> regnetx-600mf /// 일단 regnet 보류 ... 


class QuantRegNetBottleneck(QuantizedBlock):
    """
    Implementation of Quantized Bottleneck Block used in RegNet (X, Y) models.
    """

    def __init__(
        self, orig_module: RegNetBottleneck, config, qoutput=True, out_mode="calib"
    ):
        super().__init__()
        self.out_mode = out_mode
        self.qoutput = qoutput

        # Copy over attributes from the original module
        self.conv1_low = orig_module.conv1
        self.conv1_med = orig_module.conv1
        self.conv1_high = orig_module.conv1


        # self.conv1.conv = QuantizedLayer(self.conv1.conv, orig_module.act3, config)
        self.conv1_low.conv = QuantizedLayer(self.conv1.conv, orig_module.act3, config, w_qconfig=config.quant.w_qconfig_low),
        self.conv1_med.conv = QuantizedLayer(self.conv1.conv, orig_module.act3, config, w_qconfig=config.quant.w_qconfig_med),
        self.conv1_high.conv = QuantizedLayer(self.conv1.conv, orig_module.act3, config, w_qconfig=config.quant.w_qconfig_high),

        self.conv2_low = orig_module.conv2
        self.conv2_med = orig_module.conv2
        self.conv2_high = orig_module.conv2

        self.conv2_low.conv = QuantizedLayer(self.conv2.conv, orig_module.act3, config, w_qconfig=config.quant.w_qconfig_low),
        self.conv2_med.conv = QuantizedLayer(self.conv2.conv, orig_module.act3, config, w_qconfig=config.quant.w_qconfig_med),
        self.conv2_high.conv = QuantizedLayer(self.conv2.conv, orig_module.act3, config, w_qconfig=config.quant.w_qconfig_high),

        
        self.conv3_low = orig_module.conv3        
        self.conv3_med = orig_module.conv3
        self.conv3_high = orig_module.conv3

        self.conv3_low.conv = QuantizedLayer(self.conv3.conv, None, config, w_qconfig=config.quant.w_qconfig_low, qoutput=False),
        self.conv3_med.conv = QuantizedLayer(self.conv3.conv, None, config, w_qconfig=config.quant.w_qconfig_med, qoutput=False),
        self.conv3_high.conv = QuantizedLayer(self.conv3.conv, None, config, w_qconfig=config.quant.w_qconfig_high, qoutput=False),

        self.se = orig_module.se
        self.downsample= orig_module.downsample


        self.drop_path = orig_module.drop_path
        self.act3 = orig_module.act3

        # Handle downsample layer
        if self.downsample is not None:
            if hasattr(self.downsample, "conv"):
                self.downsample.conv = QuantizedLayer(
                    self.downsample.conv, None, config, w_qconfig=config.quant.w_qconfig_high, qoutput=False
                )

        # The rest of your quantization code remains the same
        if self.qoutput:

            self.block_post_act_fake_quantize_med = Quantizer(
                None, config.quant.a_qconfig_med
            )

            self.f_l = None
            self.f_m = None
            self.f_h = None
            self.f_lmh = None

            self.lambda1 = config.quant.ptmq.lambda1
            self.lambda2 = config.quant.ptmq.lambda2
            self.lambda3 = config.quant.ptmq.lambda3

            self.mixed_p = config.quant.ptmq.mixed_p

    def forward(self, x):

        shortcut = x
        x_low = self.conv1_low(x)
        x_med = self.conv1_med(x)
        x_high = self.conv1_high(x)

        x_low = self.conv2_low(x_low)
        x_med = self.conv2_low(x_med)
        x_high = self.conv2_low(x_high)


        x_low = self.se(x_low)
        x_med = self.se(x_med)
        x_high = self.se(x_high)

        x_low = self.conv3_low(x_low)
        x_med = self.conv3_med(x_med)
        x_high = self.conv3_high(x_high)

        if self.downsample is not None:
            x_low = self.drop_path(x_low) + self.downsample(shortcut)
            x_med = self.drop_path(x_med) + self.downsample(shortcut)
            x_high = self.drop_path(x_high) + self.downsample(shortcut)

        x_low = self.act3(x_low)
        x_med = self.act3(x_med)
        x_high = self.act3(x_high)

        if self.qoutput:
            if self.out_mode == "calib":
                self.f_l = self.block_post_act_fake_quantize_med(x_low)
                self.f_m = self.block_post_act_fake_quantize_med(x_med)
                self.f_h = self.block_post_act_fake_quantize_med(x_high)

                self.f_lmh = (
                    self.lambda1 * self.f_l
                    + self.lambda2 * self.f_m
                    + self.lambda3 * self.f_h
                )
                f_mixed = torch.where(torch.rand_like(x_med) < self.mixed_p, x_med, self.f_lmh)

                x = f_mixed
            elif self.out_mode == "low":
                x = self.block_post_act_fake_quantize_med(x_low)
            elif self.out_mode == "med":
                x = self.block_post_act_fake_quantize_med(x_med)
            elif self.out_mode == "high":
                x = self.block_post_act_fake_quantize_med(x_high)
            else:
                raise ValueError(
                    f"Invalid out_mode '{self.out_mode}': only ['low', 'med', 'high'] are supported"
                )
        return x


class QuantInvertedResidual(QuantizedBlock):
    def __init__ (
            self, orig_module: InvertedResidual, config, qoutput = True, out_mode = "calib"
    ):
        super().__init__()
        self.out_mode = out_mode 
        self.qoutput = qoutput
        self.use_res_connect = orig_module.use_res_connect

       
        if orig_module.expand_ratio == 1:
            self.conv_low = nn.Sequential(
                QuantizedLayer(orig_module.conv[0], orig_module.conv[2], config, w_qconfig=config.quant.w_qconfig_low),
                QuantizedLayer(orig_module.conv[3], None, config, w_qconfig=config.quant.w_qconfig_low, qoutput=False),
            )
            self.conv_med = nn.Sequential(
                QuantizedLayer(orig_module.conv[0], orig_module.conv[2], config, w_qconfig=config.quant.w_qconfig_med),
                QuantizedLayer(orig_module.conv[3], None, config, w_qconfig=config.quant.w_qconfig_med, qoutput=False),
            )
            self.conv_high = nn.Sequential(
                QuantizedLayer(orig_module.conv[0], orig_module.conv[2], config, w_qconfig=config.quant.w_qconfig_high),
                QuantizedLayer(orig_module.conv[3], None, config, w_qconfig=config.quant.w_qconfig_high, qoutput=False),
            )
        else:
            self.conv_low = nn.Sequential(
                QuantizedLayer(orig_module.conv[0], orig_module.conv[2], config, w_qconfig=config.quant.w_qconfig_low),
                QuantizedLayer(orig_module.conv[3], orig_module.conv[5], config,w_qconfig=config.quant.w_qconfig_low),
                QuantizedLayer(orig_module.conv[6], None, config, w_qconfig=config.quant.w_qconfig_low, qoutput=False),
            )
            self.conv_med = nn.Sequential(
                QuantizedLayer(orig_module.conv[0], orig_module.conv[2], config, w_qconfig=config.quant.w_qconfig_med),
                QuantizedLayer(orig_module.conv[3], orig_module.conv[5], config, w_qconfig=config.quant.w_qconfig_med),
                QuantizedLayer(orig_module.conv[6], None, config, w_qconfig=config.quant.w_qconfig_med, qoutput=False),
            )     
            self.conv_high = nn.Sequential(
                QuantizedLayer(orig_module.conv[0], orig_module.conv[2], config, w_qconfig=config.quant.w_qconfig_high),
                QuantizedLayer(orig_module.conv[3], orig_module.conv[5], config, w_qconfig=config.quant.w_qconfig_high),
                QuantizedLayer(orig_module.conv[6], None, config, w_qconfig=config.quant.w_qconfig_high, qoutput=False),
            )

        if self.qoutput:

                self.block_post_act_fake_quantize_med = Quantizer(
                    None, config.quant.a_qconfig_med
                )

                self.f_l, self.f_m, self.f_h, self.f_lmh = None, None, None, None
                self.lambda1, self.lambda2, self.lambda3 = config.quant.ptmq.lambda1,  config.quant.ptmq.lambda2,  config.quant.ptmq.lambda3
                self.mixed_p = config.quant.ptmq.mixed_p

        else:
                self.block_post_act_fake_quantize_med = None
                self.f_l, self.f_m, self.f_h, self.f_lmh = None, None, None, None
                self.lambda1, self.lambda2, self.lambda3 = config.quant.ptmq.lambda1,  config.quant.ptmq.lambda2,  config.quant.ptmq.lambda3
                self.mixed_p = config.quant.ptmq.mixed_p



    def forward(self, x):
        if self.use_res_connect:
            # x = x+self.conv(x)
            out_low = x + self.conv_low(x)
            out_mid = x + self.conv_med(x)
            out_high = x + self.conv_high(x)

        else:
            #x = self.conv(x)
            out_low = self.conv_low(x)
            out_mid = self.conv_med(x)
            out_high = self.conv_high(x)

        if self.qoutput:
            if self.out_mode == "calib":
                
                self.f_l = self.block_post_act_fake_quantize_med(out_low)
                self.f_m = self.block_post_act_fake_quantize_med(out_mid)
                self.f_h = self.block_post_act_fake_quantize_med(out_high)

                self.f_lmh = (  self.lambda1 * self.f_l
                    + self.lambda2 * self.f_m
                    + self.lambda3 * self.f_h)
                
                f_mixed = torch.where(torch.rand_like(out_mid)<self.mixed_p, out_mid, self.f_lmh)

                x = f_mixed 

            elif self.out_mode == "low":
                x = self.block_post_act_fake_quantize_med(out_low)
            elif self.out_mode == "med":
                x = self.block_post_act_fake_quantize_med(out_mid)
            elif self.out_mode == "high":
                x= self.block_post_act_fake_quantize_med(out_high)
            else: 
                raise ValueError(
                    f"Invalid out_mode '{self.out_mode}': only ['low', 'med', 'high'] are supported"
                )
        return x


def get_parent_and_name(root, full_name):
    names = full_name.split(".")
    parent = root
    for name in names[:-1]:
        parent = getattr(parent, name)
    return parent, names[-1]

# For ViT (ViT and DeiT)
class QuantVITBlock(QuantizedBlock):
    """
    Implementation of Quantized Vision Transformer Block used in Vision Transformer.
    """

    def __init__(self, orig_module: VITBlock, config, qoutput=True, out_mode="calib"):
        super().__init__()
        self.out_mode = out_mode
        self.qoutput = qoutput

        self.norm1 = orig_module.norm1

        # quantize Linear layers in Attention
        attn_linear = []
        self.attn = orig_module.attn
        # for name, module in self.attn.named_modules():
        #     if isinstance(module, nn.Linear):
        #         attn_linear.append((name, module))
        # for name, module in attn_linear:
        #     parent_module, last_name = get_parent_and_name(
        #         self.attn, name
        #     )  # Helper function to get parent module
        #     setattr(
        #         parent_module,
        #         last_name,
        #         QuantizedLayer(module, None, config, qoutput=False),
        #     )
        self.attn.qkv = QuantizedLayer(orig_module.attn.qkv, None, config, qoutput=False)
        self.attn.proj = QuantizedLayer(orig_module.attn.proj, None, config, qoutput=False)

        self.ls1 = orig_module.ls1
        self.drop_path1 = orig_module.drop_path1

        self.norm2 = orig_module.norm2

        # self.mlp = nn.Sequential(
        #     QuantizedLayer(orig_module.mlp.fc1, orig_module.mlp.act, config),
        #     orig_module.mlp.drop1,
        #     orig_module.mlp.norm,
        #     QuantizedLayer(orig_module.mlp.fc2, None, config, qoutput=False),
        #     orig_module.mlp.drop2,
        # )
        # quantize Linear layers in MLP block
        mlp_linear = []
        self.mlp = orig_module.mlp
        # for name, module in self.mlp.named_modules():
        #     if isinstance(module, nn.Linear):
        #         mlp_linear.append((name, module))
        # for name, module in mlp_linear:
        #     parent_module, last_name = get_parent_and_name(
        #         self.mlp, name
        #     )  # Helper function to get parent module
        #     if "fc2" in name:
        #         setattr(parent_module, last_name, QuantizedLayer(module, None, config, qoutput=False))  # Replace Linear layer
        #     elif "fc1" in name:
        #         setattr(parent_module, last_name, QuantizedLayer(module, None, config))
        self.mlp.fc1 = QuantizedLayer(orig_module.mlp.fc1, orig_module.mlp.act, config)
        self.mlp.fc2 = QuantizedLayer(orig_module.mlp.fc2, None, config, qoutput=False)
        self.mlp.act = nn.Identity()
        
        self.ls2 = orig_module.ls2
        self.drop_path2 = orig_module.drop_path2

        if self.qoutput:
            self.block_post_act_fake_quantize_low = Quantizer(
                None, config.quant.a_qconfig_low
            )
            self.block_post_act_fake_quantize_med = Quantizer(
                None, config.quant.a_qconfig_med
            )
            self.block_post_act_fake_quantize_high = Quantizer(
                None, config.quant.a_qconfig_high
            )

            self.f_l = None
            self.f_m = None
            self.f_h = None
            self.f_lmh = None

            self.lambda1 = config.quant.ptmq.lambda1
            self.lambda2 = config.quant.ptmq.lambda2
            self.lambda3 = config.quant.ptmq.lambda3

            self.mixed_p = config.quant.ptmq.mixed_p

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

        if self.qoutput:
            if self.out_mode == "calib":
                self.f_l = self.block_post_act_fake_quantize_low(x)
                self.f_m = self.block_post_act_fake_quantize_med(x)
                self.f_h = self.block_post_act_fake_quantize_high(x)

                self.f_lmh = (
                    self.lambda1 * self.f_l
                    + self.lambda2 * self.f_m
                    + self.lambda3 * self.f_h
                )
                f_mixed = torch.where(torch.rand_like(x) < self.mixed_p, x, self.f_lmh)

                x = f_mixed
            elif self.out_mode == "low":
                x = self.block_post_act_fake_quantize_low(x)
            elif self.out_mode == "med":
                x = self.block_post_act_fake_quantize_med(x)
            elif self.out_mode == "high":
                x = self.block_post_act_fake_quantize_high(x)
            else:
                raise ValueError(
                    f"Invalid out_mode '{self.out_mode}': only ['low', 'med', 'high'] are supported"
                )
        return x

quant_modules = {
    BasicBlock: QuantBasicBlock,
    Bottleneck: QuantBottleneck,
    RegNetBottleneck: QuantRegNetBottleneck,
    InvertedResidual: QuantInvertedResidual,
   # _InvertedResidual: _QuantInvertedResidual,
    VITBlock: QuantVITBlock,
}


def load_model(config):
    config["kwargs"] = config.get("kwargs", dict())

    print(config["kwargs"])
    print(config["type"])

    # Extract model name from config and load using timm
    if config["type"].startswith(
        ("vit", "deit")
    ):  # Assuming the config type uses ViT or any timm model
        model = timm.create_model(config["type"], pretrained=True)
    else:
        model = hubconf.__dict__[config["type"]](pretrained=True, **config["kwargs"])  # Added line to load CNN models from hubconf

    return model


def set_qmodel_block_aqbit(model, out_mode):
    """
    Function to set the activation bitwidth for all quantized blocks during inference
    """
    for name, module in model.named_modules():
        if isinstance(module, QuantizedBlock):
            # print(name)
            module.out_mode = out_mode

def set_qmodel_block_wqbit(model, out_mode):
    for name, module in model.named_modules():
        if isinstance(module, QuantizedBlock):
            print(name)
            module.out_mode = out_mode
