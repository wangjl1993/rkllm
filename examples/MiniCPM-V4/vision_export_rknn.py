'''
Author: jielong.wang jielong.wang@akuvox.com
Date: 2025-07-07 14:59:45
LastEditors: jielong.wang jielong.wang@akuvox.com
LastEditTime: 2025-10-17 17:41:10
FilePath: /rknn-llm/examples/MiniCPM-V2.6/vision_export_rknn.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import os
from rknn.api import RKNN
from sys import exit
import argparse
import cv2
import numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))

image_sizes= [[448, 448]]
batch_sizes = [1]
target_platform = 'rk3576'

def convert_encoder():
    rknn = RKNN(verbose=True)

    ONNX_MODEL=f"MiniCPMV4_image_encoder.onnx"
    RKNN_MODEL=ONNX_MODEL.replace(".onnx", f"_{target_platform}.rknn")

    input_shapes = [[[batch_size, 3, image_size[0], image_size[1]]] for batch_size in batch_sizes for image_size in image_sizes]
    print(input_shapes)

    # pre-process config
    print('--> Config model')
    rknn.config(target_platform=target_platform,
                mean_values=[128., 128., 128.], std_values=[128., 128., 128.]) # mean_values=[0.5, 0.5, 0.5], std_values=[0.5, 0.5, 0.5],
    print('done')

    # Load ONNX model
    print("--> Loading model")
    ret = rknn.load_onnx(
        model=ONNX_MODEL,
    )

    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=False, rknn_batch_size=None)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # export
    print('--> Export RKNN model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export RKNN model failed!')
        exit(ret)
    print('done')
    # rknn.init_runtime(target=target_platform)
    # # image embedding
    # img_path = "test.jpg"

    # normalize_mean = [0.5, 0.5, 0.5]
    # normalize_std = [0.5, 0.5, 0.5]

    # img = cv2.imread(img_path)
    # img = cv2.resize(img, (448, 448))
    # # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = img.astype(np.float32)
    # # img = (img - normalize_mean) / normalize_std
    # img = img[np.newaxis, :, :, :]
    # img = img.transpose(0, 3, 1, 2)
    # np.save("img.npy", img)
    # rknn.accuracy_analysis(inputs=["img.npy"], target=target_platform)
# usage: python convert_rknn.py encoder|all

if __name__ == "__main__":
    convert_encoder()
