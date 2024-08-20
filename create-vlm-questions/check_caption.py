from agi.utils.chatbot_utils import DecodingArguments, ChatBot
import os
import json
from tqdm import tqdm
import random
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, AutoModelForVision2Seq, AutoTokenizer, AutoImageProcessor, StoppingCriteria
import torch

PROMPT_PATH = "prompts/caption.txt"
DATASET_FOLDER = "/data3/dataset/VLM/object365/"
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
                    '<|system|>\nA chat between a curious user and an artificial intelligence assistant. '
                    "The assistant gives helpful, detailed, and polite answers to the user's questions.<|end|>\n"
                    f'<|user|>\n{prompt}<|end|>\n<|assistant|>\n'
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
    if model_type == "llava":
        decoding_args = DecodingArguments(
            max_tokens=2048,
            n=1,
            temperature=0.2,
            image_detail="auto",
        )
        chatbot = ChatBot(model_name)
    elif model_type == "phi3":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            trust_remote_code=True,
            torch_dtype="auto",
            _attn_implementation="eager",
        )
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        messages = [
            {"role": "user", "content": "<|image_1|>\n" + caption_prompt.replace("<image>\n", "")},
        ]
        prompt = processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

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
            if model_type == "llava":
                messages = {"text": caption_prompt, "images": [image]}
                for _ in range(10):
                    response = ""
                    for i in range(7):
                        decoding_args = DecodingArguments(
                            max_tokens=2048,
                            n=1,
                            temperature=0.2 + 0.1 * i,
                            image_detail="auto",
                        )
                        response = chatbot.call_model(
                            messages, decoding_args=decoding_args, return_list=False
                        ).strip()
                        if response:
                            break
                    response_list.append(response)
            elif model_type == "phi3":
                inputs = processor(prompt, [image], return_tensors="pt").to(device)
                for _ in range(10):
                    response = ""
                    for i in range(7):
                        generation_args = {
                            "max_new_tokens": 500,
                            "temperature": 0.2 + 0.1 * i,
                            "do_sample": False,
                        }

                        generate_ids = model.generate(
                            **inputs,
                            eos_token_id=processor.tokenizer.eos_token_id,
                            **generation_args,
                        )

                        generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
                        response = processor.batch_decode(
                            generate_ids,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        )[0]

                        if response:
                            break
                    response_list.append(response)
                
        if not response:
            error_response += 1
            progress_bar.set_description(
                f"Errors of Path: {error_image_file_path} | Errors of response {error_response} | Processing {model_name} | Captions Collected: {len(response_list)} of Image {idx}"
            )
            continue

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
    if "Phi-3" in model_name:
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

    print(f"Creating captions for {len(selected_indices)} images using llava-1.6-vicuna-7b")
    create_caption("llava-1.6-vicuna-7b", selected_indices, model_type="llava")
    print(
        f"Creating captions for {len(selected_indices)} images using phi-3-mini-4k-instruct"
    )
    create_caption("microsoft/Phi-3-vision-128k-instruct", selected_indices, model_type="phi3")


if __name__ == "__main__":
    main()
