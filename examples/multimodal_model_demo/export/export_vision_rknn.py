from rknn.api import RKNN
import numpy as np
import os
import argparse

argparse = argparse.ArgumentParser()
argparse.add_argument('--path', type=str, default='qwen2-vl-2b/qwen2_vl_2b_vision.onnx', help='model path', required=False)
argparse.add_argument('--target-platform', type=str, default='rk3588', help='target platform', required=False)
args = argparse.parse_args()

model_path = args.path
target_platform = args.target_platform

if "qwen" in model_path.lower():
    mean_value = [[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255]]
    std_value = [[0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255]]
elif 'internvl3' in model_path.lower():
    mean_value = [[0.485 * 255, 0.456 * 255, 0.406 * 255]]
    std_value = [[0.229 * 255, 0.224 * 255, 0.225 * 255]]
else:
    mean_value = [[0.5 * 255, 0.5 * 255, 0.5 * 255]]
    std_value = [[0.5 * 255, 0.5 * 255, 0.5 * 255]]

rknn = RKNN(verbose=False)
rknn.config(target_platform=target_platform, mean_values=mean_value, std_values=std_value)
rknn.load_onnx(model_path)
rknn.build(do_quantization=False, dataset=None)
os.makedirs("rknn", exist_ok=True)
rknn.export_rknn("./rknn/" + os.path.splitext(os.path.basename(model_path))[0] + "_{}.rknn".format(target_platform))
