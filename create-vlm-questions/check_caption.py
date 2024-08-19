from agi.utils.chatbot_utils import DecodingArguments, ChatBot
import argparse
import os
import json
from tqdm import tqdm
import random
from PIL import Image

PROMPT_PATH = "prompts/caption.txt"
DATASET_FOLDER = "/data3/dataset/VLM/object365/"
JSON_PATH = os.path.join(DATASET_FOLDER, "zhiyuan_objv2_train.json")


def extract_elements_with_image_id(data_dict, image_id):
    return {k: v for k, v in data_dict.items() if v.get("image_id") == image_id}


def sample_images(num_samples):
    with open(JSON_PATH, "r") as f:
        image_anns = json.load(f)  # File will be automatically closed here

    images = image_anns["images"]
    selected_indices = random.sample(range(len(images)), num_samples)
    
    return selected_indices


def create_caption(model_name, selected_indices):
    with open(JSON_PATH, "r") as f:
        image_anns = json.load(f)  # File will be automatically closed here

    images = image_anns["images"]
    anns = image_anns["annotations"]

    with open(PROMPT_PATH, "r") as f:
        caption_prompt = f.read()  # File will be automatically closed here

    decoding_args = DecodingArguments(
        max_tokens=2048,
        n=1,
        temperature=0.3,
        image_detail="auto",
    )
    chatbot = ChatBot(model_name)

    error_image_file = 0
    captions_image = {}

    progress_bar = tqdm(selected_indices, desc=f"Process Image Caption using {model_name}")

    for idx in progress_bar:
        image_info = images[idx]
        image_file = os.path.join(
            DATASET_FOLDER, image_info["file_name"].replace("images/v2/", "images/train/")
        )
        image_ann = extract_elements_with_image_id(anns, image_info["id"])

        if not os.path.exists(image_file):
            error_image_file += 1
            progress_bar.set_description(f"Errors: {error_image_file} | Processing {model_name}")
            continue

        with Image.open(image_file) as image:
            image = image.convert("RGB")
            messages = {"text": caption_prompt, "images": [image]}
            response = chatbot.call_model(
                messages, decoding_args=decoding_args, return_list=False
            ).strip()

        if response == "":
            error_image_file += 1
            progress_bar.set_description(f"Errors: {error_image_file} | Processing {model_name}")
            continue
            
        image_info_without_id = {k: v for k, v in image_info.items() if k != "id"}
        image_ann_without_image_id = {
            k: {sub_k: sub_v for sub_k, sub_v in v.items() if sub_k != "image_id"}
            for k, v in image_ann.items()
        }
        
        captions_image[image_info["id"]] = {
            "image_info": image_info_without_id,
            "image_ann": image_ann_without_image_id,
            "caption": response,
        }

        progress_bar.set_description(f"Errors: {error_image_file} | Processing {model_name}")

    if not os.path.exists("./captions"):
        os.makedirs("./captions")
    output_file_path = os.path.join("./captions", f"{model_name}_caption.json")
    with open(output_file_path, "w") as output_file:
        json.dump(captions_image, output_file, indent=4)
    print(f"Output file saved to {output_file_path}")

def main():
    selected_indices = sample_images(200)
    print(f"Creating captions for {len(selected_indices)} images using llava-1.6-vicuna-7b")
    create_caption("llava-1.6-vicuna-7b", selected_indices)
    print(f"Creating captions for {len(selected_indices)} images using phi-3-mini-4k-instruct")
    create_caption("phi-3-mini-4k-instruct", selected_indices)

if __name__ == "__main__":
    main()
