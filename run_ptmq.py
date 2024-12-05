import torch
import torch.nn as nn
import torchvision
import numpy as np
import timm

import time
import copy
import logging
import argparse
from tqdm import tqdm
import warnings
import cProfile
import pstats
import os

# disable wandb
# os.environ["WANDB_MODE"] = "disabled"

# Suppress the specific deprecation warning
warnings.filterwarnings("ignore", message="_aminmax is deprecated")


import utils
import utils.eval_utils as eval_utils
from utils.ptmq_recon import ptmq_reconstruction
from utils.fold_bn import search_fold_and_remove_bn, StraightThrough
from model import quant_modules, load_model, set_qmodel_block_wqbit
from quant.quant_state import (
    enable_calib_without_quant,
    enable_quantization,
    disable_all,
)
from quant.quant_module import QuantizedLayer, QuantizedBlock
from quant.fake_quant import QuantizeBase
from quant.observer import ObserverBase

logger = logging.getLogger("ptmq")
torch.set_float32_matmul_precision("high")


def quantize_model(model, config):
    def replace_module(module, config, qoutput=True):
        children = list(iter(module.named_children()))
        ptr, ptr_end = 0, len(children)
        prev_qmodule = None 

        while ptr < ptr_end:
            tmp_qoutput = qoutput if ptr == ptr_end - 1 else True
            name, child_module = children[ptr][0], children[ptr][1]

            if (
                type(child_module) in quant_modules
            ):  # replace blocks with quantized blocks
                setattr(
                    module,
                    name,
                    quant_modules[type(child_module)](
                        child_module, config, tmp_qoutput
                    ),
                )
            elif isinstance(child_module, (nn.Conv2d, nn.Linear)):
                setattr(
                    module,
                    name,
                    QuantizedLayer(child_module, None, config, w_qconfig=config.quant.w_qconfig,  qoutput=tmp_qoutput),
                )
                prev_qmodule = getattr(module, name)
            elif isinstance(child_module, (nn.ReLU, nn.ReLU6, nn.GELU)):
                if prev_qmodule is not None:
                    if isinstance(prev_qmodule, StraightThrough):
                        prev_qmodule = prev_qmodule.prev_module
                    prev_qmodule.activation = child_module
                    setattr(module, name, StraightThrough())
                else:
                    pass
            elif isinstance(child_module, StraightThrough):
                pass
            else:
                replace_module(child_module, config, tmp_qoutput)
            ptr += 1

    # we replace all layers to be quantized with quantization-ready layers
    replace_module(model, config, qoutput=False)

    # print("\n\nNAMED_MODULES\n\n")
    # for name, module in model.named_modules():
    #     print(name, type(module))

    # print("\n\nNAMED_CHILDREN\n\n")
    # for name, module in model.named_children():
    #     print(name, type(module))

    # for name, module in model.named_modules():
    #     if isinstance(module, QuantizedBlock):
    #         print(name, module.out_mode)

    # we store all modules in the quantized model (weight_module or activation_module)
    w_list, a_list = [], []
    for name, module in model.named_modules():
        if isinstance(module, QuantizeBase):
            if "weight" in name:
                w_list.append(module)
            elif "act" in name:
                a_list.append(module)

    # set first and last layer to 8-bit
    w_list[0].set_bit(8)
    w_list[-1].set_bit(8)

    # set the last layer's output to 8-bit
    a_list[-1].set_bit(8)

    logger.info(f"Finished quantizing model: {str(model)}")

    return model


def get_calib_data(train_loader, num_samples):
    calib_data = []
    for batch in train_loader:
        calib_data.append(batch[0])
        if len(calib_data) * batch[0].size(0) >= num_samples:
            break
    return torch.cat(calib_data, dim=0)[:num_samples]


def main(config_path):
    # get config for applying ptmq
    config = eval_utils.parse_config(config_path)
    eval_utils.set_seed(config.process.seed)
    
    if args.model:
        config.model.type = args.model
    if args.w_bit:
        config.quant.w_qconfig.bit = args.w_bit
    if args.a_bit:
        config.quant.a_qconfig.bit = args.a_bit
    if args.a_bit_low:
        config.quant.a_qconfig_low.bit = args.a_bit_low
    if args.a_bit_med:
        config.quant.a_qconfig_med.bit = args.a_bit_med
    if args.a_bit_high:
        config.quant.a_qconfig_high.bit = args.a_bit_high
    if args.scale_lr:
        config.quant.recon.scale_lr = args.scale_lr
    if args.recon_iter:
        config.quant.recon.iters = args.recon_iter
    if args.observer:
        config.quant.a_qconfig.observer = args.observer
        config.quant.a_qconfig_low.observer = args.observer
        config.quant.a_qconfig_med.observer = args.observer
        config.quant.a_qconfig_high.observer = args.observer
    
    print(f"Model: {config.model.type}")
    print(f"W{config.quant.w_qconfig_low.bit}{config.quant.w_qconfig_med.bit}{config.quant.w_qconfig_high.bit}A{config.quant.a_qconfig_med.bit}")
    print(f"Scale learning rate: {config.quant.recon.scale_lr}")
    print(f"Reconstruction iterations: {config.quant.recon.iters}")
    print(f"Observer type: {config.quant.a_qconfig.observer}")

    train_loader, val_loader = eval_utils.load_data(config, **config.data)
    calib_data = get_calib_data(train_loader, config.quant.calibrate).cuda()

    model = load_model(config.model)  # load original model

    # print("\n"*10, "ORIGINAL MODEL", "\n"*10)
    # print(model)
    # import torchvision.models as t_models
    for n, m in model.named_modules():
        if isinstance(m, torchvision.models.mobilenetv2.InvertedResidual):
            print(n, type(m.conv), len(m.conv))
    model.cuda()
    model = model.eval()
    with torch.no_grad():  # and torch.inference_mode():
        acc1, acc5 = eval_utils.validate_model(val_loader, model)
    print(f"Top-1 accuracy: {acc1:.2f}, Top-5 accuracy: {acc5:.2f}")

    search_fold_and_remove_bn(model)  # remove+fold batchnorm layers
    # print("\n"*10, "FOLDED MODEL", "\n"*10)
    # print(model)

    # quanitze model if config.quant is defined
    if hasattr(config, "quant"):
        model = quantize_model(model, config)

    # print("\n"*10, "QUANTIZED MODEL", "\n"*10)
    # print(model)

    model.cuda()  # move model to GPU
    model.eval()  # set model to evaluation mode

    fp_model = copy.deepcopy(model)  # save copy of full precision model
    disable_all(fp_model)  # disable all quantization

    # set names for all ObserverBase modules
    # ObserverBase modules are used to store intermediate values during calibration
    for name, module in model.named_modules():
        if isinstance(module, ObserverBase):
            module.set_name(name)

    # print all modules in the model
    # print(model)

    # calibration
    print("Starting model calibration...")
    tik = time.time()

    model.eval()

    # for name, module in model.named_modules():
    #     if isinstance(module, QuantizeBase):
    #         print(name, "\t", module.observer_enabled, "\t", module.fake_quant_enabled)
    # calib_loader = torch.utils.data.DataLoader(calib_data[:256], batch_size=1, shuffle=False)
    # for data in tqdm(calib_loader, desc='calibration', total=len(calib_loader)):
    enable_calib_without_quant(model, quantizer_type="act_fake_quant")
    with torch.no_grad() and torch.inference_mode():
        model(calib_data[:256].cuda())

    # weight param calibration
    enable_calib_without_quant(model, quantizer_type="weight_fake_quant")
    with torch.no_grad() and torch.inference_mode():
        model(calib_data[:2].cuda())

    tok = time.time()

    logger.info(f"Calibration of {str(model)} took {tok - tik} seconds")
    print("Completed model calibration")

    print("Starting block reconstruction...")
    tik = time.time()
    # Block reconstruction (layer reconstruction for first & last layers)a
    if hasattr(config.quant, "recon"):
        enable_quantization(model)

        def recon_model(module, fp_module):
            for name, child_module in module.named_children():
                if isinstance(child_module, (QuantizedLayer, QuantizedBlock)):
                    # logger.info(f"Reconstructing module {str(child_module)}")
                    # print(f"Reconstructing module {str(child_module)}")
                    ptmq_reconstruction(
                        model,
                        fp_model,
                        child_module,
                        name,
                        getattr(fp_module, name),
                        calib_data,
                        config.quant,
                    )
                else:
                    recon_model(child_module, getattr(fp_module, name))

        recon_model(model, fp_model)
    tok = time.time()
    print("Completed block reconstruction")
    print(f"PTMQ block reconstruction took {tok - tik:.2f} seconds")

    w_qmodes = ["low", "med", "high"]
    a_qbit = config.quant.a_qconfig_med.bit,
    w_qbits = [config.quant.w_qconfig_low, 
               config.quant.w_qconfig_med, 
               config.quant.w_qconfig_high, 
               ]

    # # save ptmq model
    # torch.save(
    #     model.state_dict(), f"ptmq_w{w_qbits[0]}{w_qbits[1]}{w_qbits[2]}_a{a_qbit}.pth"
    # )

    enable_quantization(model)

    # for name, module in model.named_modules():
    #     if isinstance(module, QuantizeBase):
    #         print(name, "\t", module.observer_enabled, "\t", module.fake_quant_enabled)

    for w_qmode, w_qbit in zip(w_qmodes, w_qbits):
        # if a_qbit < w_qbit:
        #     continue
        
        set_qmodel_block_wqbit(model, w_qmode)

        print(
            f"Starting model evaluation of W{w_qbit}A{a_qbit} block reconstruction ({a_qmode})..."
        )
        acc1, acc5 = eval_utils.validate_model(val_loader, model)

        print(f"Top-1 accuracy: {acc1:.2f}, Top-5 accuracy: {acc5:.2f}")
        
        # if a_qbit >= w_qbit:
        #     # break
        #     pass

    # print out all scale and zero_point values
    # for name, module in model.named_modules():
    #     if isinstance(module, QuantizeBase):
    #         print(name)
    #         print(f"num_bits = {module.bit}")
    #         print(f"quant_range = [{module.quant_min}, {module.quant_max}]")
    #         print(
    #             f"max_val - min_val = {module.scale * (module.quant_max - module.quant_min)}"
    #         )
    #         print(f"scale = {module.scale}")
    #         print(f"zero_point = {module.zero_point}")
    #         print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="configuration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-c", "--config", default="config/resnet18.yaml", type=str, help="Path to config file"
    )
    parser.add_argument(
        "-m", "--model", default="resnet18", type=str, help="Model to be quantized"
    )
    parser.add_argument(
        "-w", "--w_bit", default=8, type=int, help="Weight bitwidth for quantization"
    )
    parser.add_argument(
        "-a", "--a_bit", default=8, type=int, help="Activation bitwidth for quantization"
    )
    parser.add_argument(
        "-al", "--a_bit_low", default=6, type=int, help="Activation bitwidth for quantization"
    )
    parser.add_argument(
        "-am", "--a_bit_med", default=3, type=int, help="Activation bitwidth for quantization"
    ) # 이 부분 고치기 
    parser.add_argument(
        "-ah", "--a_bit_high", default=8, type=int, help="Activation bitwidth for quantization"
    )
    parser.add_argument(
        "-lr", "--scale_lr", default=4e-5, type=float, help="Learning rate for scale"
    )
    parser.add_argument(
        "-i", "--recon_iter", default=100, type=int, help="Number of reconstruction iterations"
    )
    parser.add_argument(
        "-o", "--observer", default=None, type=str, help="Observer type for quantization"
    )
    args = parser.parse_args()

    # Start profiling
    # profiler = cProfile.Profile()
    # profiler.enable()

    main(args.config)

    # Stop profiling
    # profiler.disable()
    # profiler.dump_stats('profile_output.pstat')

    # Optionally, print profiling stats
    # stats = pstats.Stats(profiler)
    # stats.sort_stats('cumulative').print_stats(10)
