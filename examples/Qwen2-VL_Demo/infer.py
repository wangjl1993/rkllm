'''
Author: jielong.wang jielong.wang@akuvox.com
Date: 2025-07-02 17:13:21
LastEditors: jielong.wang jielong.wang@akuvox.com
LastEditTime: 2025-07-03 10:50:25
FilePath: /rknn-llm/examples/Qwen2-VL_Demo/infer.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from PIL import Image
import requests
import torch
from torchvision import io
from typing import Dict
from transformers import AutoModel, AutoProcessor, Qwen2VLForConditionalGeneration

# Load the model in half-precision on the available device(s)
path = "/home/ai_team/jielong/.cache/huggingface/hub/models--Qwen--Qwen2-VL-2B-Instruct/snapshots/895c3a49bc3fa70a340399125c650a463535e71c"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    path, torch_dtype="auto", device_map="auto",
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(path)

image = Image.open('./data/demo.jpg')
image = image.resize((392, 392), 3)
conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Preprocess the inputs
text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
# Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n'

inputs = processor(
    text=[text_prompt], images=[image], padding=True, return_tensors="pt"
)
inputs = inputs
print("inputs.pixel_values: ", inputs['pixel_values'].shape)

# Inference: Generation of the output
output_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids = [
    output_ids[len(input_ids) :]
    for input_ids, output_ids in zip(inputs.input_ids, output_ids)
]
output_text = processor.batch_decode(
    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
)
print(output_text)
