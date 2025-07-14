'''
Author: jielong.wang jielong.wang@akuvox.com
Date: 2025-07-07 16:07:50
LastEditors: jielong.wang jielong.wang@akuvox.com
LastEditTime: 2025-07-07 16:41:05
FilePath: /rknn-llm/examples/MiniCPM-V2.6/rkllm_convert.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from rkllm.api import RKLLM

modelpath = '/home/ai_team/jielong/.cache/huggingface/hub/models--openbmb--MiniCPM-V-2_6/snapshots/6c04d9e3022bcff6e6738dfb1fc19a5cfd2a855f'
llm = RKLLM()

ret = llm.load_huggingface(model=modelpath, model_lora=None, device='cpu')
if ret != 0:
    print('Load model failed!')
    exit(ret)

qparams = None
ret = llm.build(do_quantization=True, optimization_level=1, quantized_dtype='w4a16',
                quantized_algorithm='normal', target_platform='rk3576', num_npu_core=2, extra_qparams=qparams)

if ret != 0:
    print('Build model failed!')
    exit(ret)

# Export rkllm model
ret = llm.export_rkllm("./qwen_w4a16.rkllm")
if ret != 0:
    print('Export model failed!')
    exit(ret)
