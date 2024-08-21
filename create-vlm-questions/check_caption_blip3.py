import os
import json
from tqdm import tqdm
import random
from PIL import Image
from transformers import (
    AutoModelForVision2Seq,
    AutoTokenizer,
    AutoImageProcessor,
    StoppingCriteria,
)
import torch

PROMPT_PATH = "prompts/caption.txt"
DATASET_FOLDER = "/ariesdv0/zhanling/vlm-datasets/object365/"
JSON_PATH = os.path.join(DATASET_FOLDER, "zhiyuan_objv2_train.json")
INDICES_FILE = "selected_indices.json"


def extract_elements_with_image_id(anns_list, image_id):
    return [ann for ann in anns_list if ann.get("image_id") == image_id]


def sample_images(num_samples):
    with open(JSON_PATH, "r") as f:
        image_anns = json.load(f)
    images = image_anns["images"]
    selected_indices = random.sample(range(len(images)), num_samples)

    with open(INDICES_FILE, "w") as f:
        json.dump(selected_indices, f)

    return selected_indices


def load_indices():

    with open(INDICES_FILE, "r") as f:
        selected_indices = json.load(f)
    return selected_indices


def apply_prompt_template(prompt):
    s = (
        "<|system|>\nA chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions.<|end|>\n"
        f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
    )
    return s


def create_caption(model_name, selected_indices, model_type):

    with open(JSON_PATH, "r") as f:
        image_anns = json.load(f)

    images = image_anns["images"]
    anns = image_anns["annotations"]

    with open(PROMPT_PATH, "r") as f:
        caption_prompt = f.read()

    error_image_file_path = 0
    error_response = 0
    captions_image = {}

    progress_bar = tqdm(
        selected_indices, desc=f"Process Image Caption using {model_name}"
    )
    # breakpoint()

    model = AutoModelForVision2Seq.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, use_fast=False, legacy=False
    )
    image_processor = AutoImageProcessor.from_pretrained(
        model_name, trust_remote_code=True
    )
    tokenizer = model.update_special_tokens(tokenizer)
    model = model.to("cuda")
    model.eval()
    tokenizer.padding_side = "left"

    for idx in progress_bar:
        image_info = images[idx]

        if "images/v2/" in image_info["file_name"]:
            image_file = image_info["file_name"].replace("images/v2/", "images/train/")
        elif "images/v1/" in image_info["file_name"]:
            image_file = image_info["file_name"].replace("images/v1/", "images/train/")

        image_file = os.path.join(DATASET_FOLDER, image_file)
        image_ann = extract_elements_with_image_id(anns, image_info["id"])
        if not os.path.exists(image_file):
            error_image_file_path += 1
            print(f"Image file not found: {image_file}")
            progress_bar.set_description(
                f"Errors of Path: {error_image_file_path} | Errors of response {error_response} | Processing {model_name} | Captions Collected: {len(response_list)} of Image {idx}"
            )
            continue

        response_list = []
        with Image.open(image_file) as image:
            image = image.convert("RGB")
            image_list = []
            image_sizes = []
            image_list.append(
                image_processor([image], image_aspect_ratio="anyres")[
                    "pixel_values"
                ].cuda()
            )
            image_sizes.append(image.size)
            inputs = {"pixel_values": [image_list]}
            prompt = apply_prompt_template(caption_prompt.replace("<image>\n", ""))
            language_inputs = tokenizer([prompt], return_tensors="pt")
            inputs.update(language_inputs)
            for name, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    inputs[name] = value.cuda()
            for _ in range(10):
                response = ""
                for i in range(7):
                    temperature = 0.2 + 0.1 * i
                    generated_text = model.generate(
                        **inputs,
                        image_size=[image_sizes],
                        pad_token_id=tokenizer.pad_token_id,
                        temperature=temperature,
                        do_sample=False,
                        max_new_tokens=2048,
                        top_p=None,
                        num_beams=1,
                    )
                    # breakpoint()
                    response = tokenizer.decode(generated_text[0], skip_special_tokens=True).split("<|end|>")[0]
                    if response:  # Stop if a non-empty response is obtained
                        break

            if not response:
                error_response += 1
                progress_bar.set_description(
                    f"Errors of Path: {error_image_file_path} | Errors of response {error_response} | Processing {model_name} | Captions Collected: {len(response_list)} of Image {idx}"
                )
                continue
            response_list.append(response)

        image_info_without_id = {k: v for k, v in image_info.items() if k != "id"}
        image_ann_without_image_id = [
            {sub_k: sub_v for sub_k, sub_v in ann.items() if sub_k != "image_id"}
            for ann in image_ann
        ]

        captions_image[image_info["id"]] = {
            "image_info": image_info_without_id,
            "image_ann": image_ann_without_image_id,
            "caption": response_list,
        }

        progress_bar.set_description(
            f"Errors of Path: {error_image_file_path} | Errors of response {error_response} | Processing {model_name} | Captions Collected: {len(response_list)} of Image {idx}"
        )

    if not os.path.exists("./captions"):
        os.makedirs("./captions")
    model_name = model_name.split("/")[-1]
    output_file_path = os.path.join("./captions", f"{model_name}_caption.json")
    with open(output_file_path, "w") as output_file:
        json.dump(captions_image, output_file, indent=4)
    print(f"Output file saved to {output_file_path}")


def main():
    if os.path.exists(INDICES_FILE):
        selected_indices = load_indices()
        print(f"Loaded {len(selected_indices)} indices from {INDICES_FILE}")
    else:
        selected_indices = sample_images(200)
        print(f"Created new selection of {len(selected_indices)} images")
    print(f"Creating captions for {len(selected_indices)} images using blip3")
    create_caption(
        "Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5",
        selected_indices,
        model_type="xgen-mm",
    )


if __name__ == "__main__":
    main()
