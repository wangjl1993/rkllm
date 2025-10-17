'''
Author: jielong.wang jielong.wang@akuvox.com
Date: 2025-07-07 14:53:36
LastEditors: jielong.wang jielong.wang@akuvox.com
LastEditTime: 2025-10-17 17:44:05
FilePath: /rknn-llm/examples/MiniCPM-V2.6/vision_export_onnx.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/home/ai_team/jielong/.cache/huggingface/hub/models--openbmb--MiniCPM-V-4/snapshots/0968cf95dd0ed5584b5a50d0ded4ce29421674f7"
DEVICE_MAP = "cpu"

origin_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, trust_remote_code=True, attn_implementation='eager', device_map=DEVICE_MAP).eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

for param in origin_model.parameters():
    param.requires_grad = False

class VisionTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vpm = origin_model.vpm
        self.resampler = origin_model.resampler
        self.tgt_sizes = torch.Tensor([[32, 32]]).type(torch.int32)

    def forward(self, pixel_values):
        print(f"pixel_values shape: {pixel_values.shape}")
        vit_embeds = self.vpm(pixel_values).last_hidden_state
        print(f"vit_embeds shape before resampling: {vit_embeds.shape}")
        vit_embeds = self.resampler(vit_embeds, self.tgt_sizes)
        print(f"vit_embeds shape after resampling: {vit_embeds.shape}")
        return vit_embeds


def convert_vision_transformer():
    model = VisionTransformer()
    IMAGE_SIZE = 448
    pixel_values = torch.randn(
        (1, 3, IMAGE_SIZE, IMAGE_SIZE))
    
    # test first
    vit_embeds = model(pixel_values)
    print(vit_embeds.shape)  #1x64x2560
    if vit_embeds.shape != (1, 64, 2560):
        raise ValueError("vit_embeds shape is not correct, something is wrong")

    save_file = "MiniCPMV4_image_encoder_rk3576.onnx"
    torch.onnx.export(model, pixel_values,
                      save_file,
                      verbose=False,
                      input_names=['pixel_values'],
                      output_names=['vision_embeds'],
                      opset_version=18)

if __name__ == "__main__":
    convert_vision_transformer()