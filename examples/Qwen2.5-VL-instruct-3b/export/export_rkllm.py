'''
Author: jielong.wang jielong.wang@akuvox.com
Date: 2025-08-08 09:55:51
LastEditors: jielong.wang jielong.wang@akuvox.com
LastEditTime: 2025-08-08 11:19:09
FilePath: /rknn-llm/examples/Qwen2.5-VL-instruct-3b/export/export_rkllm.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
from rkllm.api import RKLLM
from datasets import load_dataset
from transformers import  AutoTokenizer
from tqdm import tqdm
import torch
from torch import nn
import argparse

argparse = argparse.ArgumentParser()
argparse.add_argument('--path', type=str, default='/home/ai_team/jielong/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3', help='model path', required=False)
argparse.add_argument('--target-platform', type=str, default='rk3576', help='target platform', required=False)
argparse.add_argument('--num_npu_core', type=int, default=2, help='npu core num', required=False)
argparse.add_argument('--quantized_dtype', type=str, default='w4a16', help='quantized dtype', required=False)
argparse.add_argument('--device', type=str, default='cpu', help='device', required=False)
argparse.add_argument('--savepath', type=str, default='qwen2.5_vl_3b_instruct.rkllm', help='save path', required=False)
args = argparse.parse_args()

modelpath = args.path
target_platform = args.target_platform
num_npu_core = args.num_npu_core
quantized_dtype = args.quantized_dtype
savepath = args.savepath
llm = RKLLM()

# Load model
# Use 'export CUDA_VISIBLE_DEVICES=2' to specify GPU device
ret = llm.load_huggingface(model=modelpath, device=args.device)
if ret != 0:
    print('Load model failed!')
    exit(ret)

# Build model
dataset = None #'data/inputs.json'

qparams = None
ret = llm.build(do_quantization=True, optimization_level=1, quantized_dtype=quantized_dtype,
                quantized_algorithm='normal', target_platform=target_platform, num_npu_core=num_npu_core, extra_qparams=qparams, dataset=dataset)

if ret != 0:
    print('Build model failed!')
    exit(ret)

# # Export rkllm model
ret = llm.export_rkllm(savepath)
if ret != 0:
    print('Export model failed!')
    exit(ret)


