import torch
import torchvision
import torch.nn as nn
import timm
import gc


from torchvision.models.resnet import BasicBlock as ResNetBasicBlock
from torchvision.models.resnet import Bottleneck as ResNetBottleneck
from torchvision.models.mobilenetv2 import InvertedResidual, mobilenet_v2
from timm.models.regnet import Bottleneck as RegNetBottleneck
from timm.models.vision_transformer import Block as VITBlock
from quant.quant_module import QuantizedLayer, QuantizedBlock, Quantizer


#from models.regnet import Bottleneck as RegNetBottleneck
# 1. QuantBasicBlock -> resnet18 

class QuantBasicBlock(QuantizedBlock): 
    """
    Implementation of Quantized BasicBlock used in ResNet-18 and ResNet-34.
    qoutput:    whether to quantize the block output
    out_mode:   setting inference block output mode
        - "mixed":  mixed feature output. used only for block reconstruction
        - "low":    low bit feature output
        - "med":    medium bit feature output
        - "high":   high bit feature output
    """
    def __init__(self, orig_module: ResNetBasicBlock, config, qoutput=True, out_mode="calib"):
        super().__init__()
        self.out_mode = out_mode
        self.qoutput = qoutput


        # weight 에 대한 quantizer 
       
        self.conv1_relu_low = QuantizedLayer(orig_module.conv1, orig_module.relu1, config, w_qconfig=config.quant.w_qconfig_low, qoutput=False)
        self.conv1_relu_mid = QuantizedLayer(orig_module.conv1, orig_module.relu1, config, w_qconfig=config.quant.w_qconfig_med, qoutput=False)
        self.conv1_relu_high = QuantizedLayer(orig_module.conv1, orig_module.relu1, config, w_qconfig=config.quant.w_qconfig_high, qoutput=False)

        self.conv2_low = QuantizedLayer(orig_module.conv2, None, config,  w_qconfig=config.quant.w_qconfig_low, qoutput=False)
        self.conv2_mid = QuantizedLayer(orig_module.conv2, None, config, w_qconfig=config.quant.w_qconfig_med, qoutput=False)
        self.conv2_high = QuantizedLayer(orig_module.conv2, None, config, w_qconfig=config.quant.w_qconfig_high, qoutput=False)
        
        if orig_module.downsample is None:
            self.downsample_low = None
            self.downsample_mid = None
            self.downsample_high = None
        else:
            self.downsample_low = QuantizedLayer(orig_module.downsample[0], None, config, w_qconfig=config.quant.w_qconfig_low, qoutput=False)
            self.downsample_mid= QuantizedLayer(orig_module.downsample[0], None, config, w_qconfig=config.quant.w_qconfig_med, qoutput=False)
            self.downsample_high= QuantizedLayer(orig_module.downsample[0], None, config, w_qconfig=config.quant.w_qconfig_high, qoutput=False)

        self.activation = orig_module.relu2

        if self.qoutput:
            # self.block_post_act_fake_quantize_low = Quantizer(None, config.quant.a_qconfig_low)
            self.block_post_act_fake_quantize_med = Quantizer(None, config.quant.a_qconfig_med)
            # self.block_post_act_fake_quantize_high = Quantizer(None, config.quant.a_qconfig_high)
            
            self.f_l = None
            self.f_m = None
            self.f_h = None
            self.f_lmh = None
            
            self.lambda1 = config.quant.ptmq.lambda1
            self.lambda2 = config.quant.ptmq.lambda2
            self.lambda3 = config.quant.ptmq.lambda3
            self.mixed_p = config.quant.ptmq.mixed_p
            
    def forward(self, x):
        residual_low = x if self.downsample_low is None else self.downsample_low(x)
        residual_mid = x if self.downsample_mid is None else self.downsample_mid(x)
        residual_high = x if self.downsample_high is None else self.downsample_high(x)

        # Conv1 -> low, mid, high bit-width로 양자화된 출력 계산 
        
        out_low = self.conv1_relu_low(x)
        out_mid = self.conv1_relu_mid(x)
        out_high = self.conv1_relu_high(x)

        # Conv2 -> low, mid, hight 비트로 양자화된 출력 계산 
        out_low = self.conv2_low(out_low)
        out_mid = self.conv2_mid(out_mid)
        out_high = self.conv2_high(out_high)

        out_low += residual_low
        out_mid += residual_mid
        out_high += residual_high

          # 활성화 함수 적용
        out_low = self.activation(out_low)
        out_mid = self.activation(out_mid)
        out_high = self.activation(out_high)


        if self.qoutput:
            if self.out_mode == "calib":
   
                self.f_l = self.block_post_act_fake_quantize_med(out_low)
                self.f_m = self.block_post_act_fake_quantize_med(out_mid)
                self.f_h = self.block_post_act_fake_quantize_med(out_high)
                
                self.f_lmh = self.lambda1 * self.f_l + self.lambda2 * self.f_m + self.lambda3 * self.f_h
                f_mixed = torch.where(torch.rand_like(out_mid) < self.mixed_p, out_mid, self.f_lmh)
                
                out = f_mixed
                
            elif self.out_mode == "low":
               # out = self.block_post_act_fake_quantize_low(out)
               out = self.block_post_act_fake_quantize_med(out_low)
            elif self.out_mode == "med":
               # out = self.block_post_act_fake_quantize_med(out)
               out= self.block_post_act_fake_quantize_med(out_mid)
            elif self.out_mode == "high":
               # out = self.block_post_act_fake_quantize_high(out)
               out = self.block_post_act_fake_quantize_med(out_high)
            else:
                raise ValueError(f"Invalid out_mode '{self.out_mode}': only ['low', 'med', 'high'] are supported")
        return out

# 2. QuantBottleneck -> resnet-50 

class QuantBottleneck(QuantizedBlock):
    """
    Implementation of Quantized Bottleneck Block used in ResNet-50, Resnet-101, and ResNet-152.
    """
    # weight multi bit 은 차후에
    def __init__(self, orig_module: ResNetBottleneck, config, qoutput=True, out_mode="calib"):
        super().__init__()
        self.out_mode = out_mode

        self.qoutput = qoutput

        self.conv1_relu_low = QuantizedLayer(orig_module.conv1, orig_module.relu1, config, w_qconfig=config.quant.w_qconfig_low, qoutput=False)
        self.conv1_relu_mid = QuantizedLayer(orig_module.conv1, orig_module.relu1, config, w_qconfig=config.quant.w_qconfig_med, qoutput=False)
        self.conv1_relu_high = QuantizedLayer(orig_module.conv1, orig_module.relu1, config, w_qconfig=config.quant.w_qconfig_high, qoutput=False)

        self.conv2_relu_low = QuantizedLayer(orig_module.conv2, orig_module.relu2, config, w_qconfig= config.quant.w_qconfig_low, qoutput=False)
        self.conv2_relu_mid = QuantizedLayer(orig_module.conv2, orig_module.relu2, config, w_qconfig= config.quant.w_qconfig_med, qoutput=False)
        self.conv2_relu_high = QuantizedLayer(orig_module.conv2, orig_module.relu2, config, w_qconfig= config.quant.w_qconfig_high, qoutput=False)

        self.conv3_low = QuantizedLayer(orig_module.conv3, None, config, w_qconfig=config.quant.w_qconfig_low, qoutput=False)
        self.conv3_mid = QuantizedLayer(orig_module.conv3, None, config, w_qconfig=config.quant.w_qconfig_med, qoutput=False)
        self.conv3_high = QuantizedLayer(orig_module.conv3, None, config, w_qconfig = config.quant.w_qconfig_high, qoutput=False)

        
        if orig_module.downsample is None:
            self.downsample_low = None
            self.downsample_mid = None
            self.downsample_high = None
 
        else:
            self.downsample_low = QuantizedLayer(orig_module.downsample[0], None, config, config.quant.w_qconfig_low, qoutput=False)
            self.downsample_mid = QuantizedLayer(orig_module.downsample[0], None, config, config.quant.w_qconfig_med, qoutput=False)
            self.downsample_high = QuantizedLayer(orig_module.downsample[0], None, config, config.quant.w_qconfig_high, qoutput=False)

        self.activation = orig_module.relu3
        if self.qoutput:
            #self.block_post_act_fake_quantize = Quantizer(None, config.quant.a_qconfig)
            # self.block_post_act_fake_quantize_low = Quantizer(None, config.quant.a_qconfig_low)
            self.block_post_act_fake_quantize_med = Quantizer(None, config.quant.a_qconfig_med)
            # self.block_post_act_fake_quantize_high = Quantizer(None, config.quant.a_qconfig_high)
            self.f_l = None 
            self.f_m = None 
            self.f_h = None 
            self.f_lmh = None 

            self.lambda1 = config.quant.ptmq.lambda1
            self.lambda2 = config.quant.ptmq.lambda2
            self.lambda3 = config.quant.ptmq.lambda3

            self.mixed_p = config.quant.ptmq.mixed_p 



    def forward(self, x):
        residual_low = x if self.downsample_low is None else self.downsample_low(x)
        residual_mid = x if self.downsample_mid is None else self.downsample_mid(x)
        residual_high = x if self.downsample_high is None else self.downsample_high(x)


        out_low = self.conv1_relu_low(x)
        out_mid = self.conv1_relu_mid(x)
        out_high = self.conv1_relu_high(x)

        out_low = self.conv2_relu_low(out_low)
        out_mid = self.conv2_relu_mid(out_mid)
        out_high = self.conv2_relu_high(out_high)

        out_low = self.conv3_low(out_low)
        out_mid = self.conv3_mid(out_mid)
        out_high = self.conv3_high(out_high)


        out_low += residual_low
        out_mid += residual_mid
        out_high += residual_high

        out_low = self.activation(out_low)
        out_mid = self.activation(out_mid)
        out_high = self.activation(out_high)


        if self.qoutput:
            if self.out_mode == "calib":
                self.f_l = self.block_post_act_fake_quantize_med(out_low)
                self.f_m = self.block_post_act_fake_quantize_med(out_mid)
                self.f_h = self.block_post_act_fake_quantize_med(out_high)

                self.f_lmh = self.lambda1*self.f_l + self.lambda2*self.f_m + self.lambda3*self.f_h

                f_mixed = torch.where(torch.rand_like(out_mid) < self.mixed_p , out_mid, self.f_lmh)

                x = f_mixed 
            elif self.out_mode == "low":
                x = self.block_post_act_fake_quantize_med(out_low)
            elif self.out_mode == "med":
                x = self.block_post_act_fake_quantize_med(out_mid)
            elif self.out_mode == "high":
                x = self.block_post_act_fake_quantize_med(out_high)

            else :
                raise ValueError(
                    f"Invalid out_mode '{self.out_mode}: only ['low','med','high'] are supported."
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



# 4. QuantInvertedResidual -> mobilenetv2

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


# 5. vit , deit
quant_modules = {
    ResNetBasicBlock: QuantBasicBlock,  # For ResNet-18 and ResNet-34
    ResNetBottleneck: QuantBottleneck,  # For ResNet-50, ResNet-101, and ResNet-152
    # RegNetBottleneck: QuantRegNetBottleneck,  # For RegNetX-600MF
    InvertedResidual: QuantInvertedResidual,  # For MobileNetV2
    # VITBlock: QuantVITBlock,  # For Vision Transformer
}




def load_model(config):
    config["kwargs"] = config.get("kwargs", dict())

    print(config["kwargs"])
    print(config["type"])

    # Extract model name from config and load using timm
    if config["type"].startswith(
        ("vit", "regnet", "deit")
    ):  # Assuming the config type uses ViT or any timm model
        model = timm.create_model(config["type"], pretrained=True)
    else:
        # Fallback for non-timm models - load from torchvision
        name = config["type"]
        model = getattr(torchvision.models, name)(weights="DEFAULT")

        # model = eval(config["type"])(**config["kwargs"])

        # # Load the model checkpoint
        # checkpoint = torch.load(config["path"], map_location="cpu", weights_only=True)
        # model.load_state_dict(checkpoint)

    return model


def set_qmodel_block_aqbit(model, out_mode):
    for name, module in model.named_modules():
        if isinstance(module, QuantizedBlock):
            # print(name)
            module.out_mode = out_mode

def set_qmodel_block_wqbit(model, out_mode):
    for name, module in model.named_modules():
        if isinstance(module, QuantizedBlock):
            print(name)
            module.out_mode = out_mode
