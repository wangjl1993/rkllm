'''
Author: jielong.wang jielong.wang@akuvox.com
Date: 2025-10-17 17:28:42
LastEditors: jielong.wang jielong.wang@akuvox.com
LastEditTime: 2025-10-17 17:33:11
FilePath: /rknn-llm/examples/MiniCPM-V4/data/make_input_embeds_for_quantize.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="Build quantization inputs for MiniCPM-V4.")
    parser.add_argument("--path", type=str, default="/home/ai_team/jielong/.cache/huggingface/hub/models--openbmb--MiniCPM-V-4/snapshots/0968cf95dd0ed5584b5a50d0ded4ce29421674f7", help="MiniCPM-V4 模型的本地目录或仓库名")
    parser.add_argument("--dataset", type=str, help="量化样本列表 JSON")
    parser.add_argument("--output-dir", type=str, default="./data",help="输出目录，保存 inputs_embeds 与 inputs.json")
    return parser.parse_args()


def load_image(image_path: Path) -> Image.Image:
    with Image.open(image_path) as img:
        return img.convert("RGB").resize((448,448))



def main():
    args = parse_args()


    model = AutoModel.from_pretrained(
        args.path,
        torch_dtype=torch.float32,
        # low_cpu_mem_usage=True,
        attn_implementation='sdpa',
        trust_remote_code=True,
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(args.path, trust_remote_code=True)

    with open(args.dataset, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    output_dir = Path(args.output_dir)
    embeds_dir = output_dir / "inputs_embeds"
    embeds_dir.mkdir(parents=True, exist_ok=True)

    records = []
    image_token_placeholder = "(<image>./</image>)"

    for sample in tqdm(dataset, desc="Embedding samples"):
        image_path = Path(sample["image_path"]) / sample["image"]
        image = load_image(image_path)

        user_prompt = f"{image_token_placeholder}\n{sample['input'].rstrip()}"
        template_messages = [{"role": "user", "content": user_prompt}]
        prompt = processor.tokenizer.apply_chat_template(
            template_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        encoded = processor(
            [prompt],
            [[image]],
            padding=True,
            return_tensors="pt",
        )
        image.close()

        model_inputs = {
            "input_ids": encoded["input_ids"],
            "pixel_values": encoded.get("pixel_values"),
            "image_bound": encoded.get("image_bound"),
            "tgt_sizes": encoded.get("tgt_sizes"),
            "position_ids": encoded.get("position_ids"),
        }

        with torch.inference_mode():
            input_embeds, _ = model.get_vllm_embedding(model_inputs)

        embed_np = input_embeds.to(dtype=torch.float16).cpu().numpy()
        name = Path(sample["image"]).stem
        np.save(embeds_dir / f"{name}.npy", embed_np)

        records.append({"input_embed": embed_np.tolist(), "target": sample["target"]})

    with open(output_dir / "inputs.json", "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print("Done")


if __name__ == "__main__":
    main()