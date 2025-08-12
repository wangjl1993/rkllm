'''
Author: jielong.wang jielong.wang@akuvox.com
Date: 2025-07-08 11:48:21
LastEditors: jielong.wang jielong.wang@akuvox.com
LastEditTime: 2025-08-11 11:03:15
FilePath: /rknn-llm/examples/Qwen2-VL_Demo/export/export_vision_rknn.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from rknn.api import RKNN
import numpy as np
import os
import argparse

argparse = argparse.ArgumentParser()
argparse.add_argument('--path', type=str, help='model path', required=False)
argparse.add_argument('--target-platform', type=str, default='rk3576', help='target platform', required=False)
args = argparse.parse_args()

model_path = args.path
target_platform = args.target_platform

rknn = RKNN(verbose=False)
rknn.config(target_platform=target_platform, mean_values=[[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255]], std_values=[[0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255]])
rknn.load_onnx(model_path)
rknn.build(do_quantization=False, dataset=None)
# os.makedirs("rknn", exist_ok=True)
rknn.export_rknn(os.path.splitext(os.path.basename(model_path))[0] + "_{}.rknn".format(target_platform))
