import sys
import os
import json
import argparse
import csv
import re
from PIL import Image
import random
from utils import read_data_from_csv
import ast

os.environ["AGI_ROOT"] = "/home/zhaoyang/projects/neural-reasoning"
sys.path.append(os.path.join(os.environ["AGI_ROOT"]))
from agi.utils.openai_utils import get_total_money
from agi.utils.chatbot_utils import DecodingArguments, ChatBot

CURRENT_FOLDER = os.path.dirname(os.path.abspath(__file__))

def get_all_objs(data_for_all_images):
    objects_for_all_images = set()
    for item in data_for_all_images:
        # item_dic =  ast.literal_eval(item['object_infor'])
        objects_for_all_images.update(item['object_infor'].keys())
    return list(objects_for_all_images)
    

def generate_obj_candidates_for_per_image(base_obj_candidates, all_objs, num_elements=50):
    needed_objs = num_elements - len(base_obj_candidates)
    available_objs_candidates = list(set(all_objs) - set(base_obj_candidates))
    selected_objs_candidates = random.sample(available_objs_candidates, min(len(available_objs_candidates), needed_objs))
    final_objs_candidates = list(base_obj_candidates + selected_objs_candidates)
    random.shuffle(final_objs_candidates)
    return final_objs_candidates


def validate_format(s):
    pattern = r"^\[('[^']*'(, '[^']*')*)?\]$"
    return bool(re.match(pattern, s))


def write_data_to_csv(filename, data):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        for row in data:
            writer.writerow(row)
            print(f'Added row: {row}')



def delete_object_infor_without_in_truth(object_infor, true_object_infor):
    for obj_name in list(object_infor.keys()):
        if obj_name.lower() not in true_object_infor:
            del object_infor[obj_name]
    return object_infor

def main(args):
    prompt_path = os.path.join(CURRENT_FOLDER, "prompts/generate_obj_categories.txt")
    prompt_template = open(prompt_path, "r").read()
    print(f"Prompt template:\n{prompt_template}")
    actor_vlm_model = ChatBot(args.vlm_model)
    image_obj_infor_path = os.path.join(CURRENT_FOLDER, args.image_obj_infor_path)
    data = read_data_from_csv(image_obj_infor_path)
    all_objs = get_all_objs(data)
    answer = []
    answer.append(['true_objs', 'predicted_objs', 'original_image_path'])
    for idx, item in enumerate(data):
        true_object_infor = list(item["object_bboxes"].keys())
        object_infor = item["object_infor"]
        object_infor = delete_object_infor_without_in_truth(object_infor, true_object_infor)
        objs_infor = list(object_infor.keys()) 
        true_objs_infor = str(objs_infor)
        obj_candidates_for_per_image = str(generate_obj_candidates_for_per_image(objs_infor, all_objs))
        # obj_candidates_for_per_image =  f"[{', '.join(obj_candidates_for_per_image)}]"
        prompt = prompt_template.format(candidates=obj_candidates_for_per_image)
        
        original_image_pth = item["original_image_path"]
        image_base_name = os.path.basename(original_image_pth)
        image = Image.open(original_image_pth)
        
        messages = {
            "text": prompt,
            "images": [image],
        }
        for tried in range(3):
            action_vlm_decoding_args = DecodingArguments(
                max_tokens=2048,
                n=1,
                temperature=0.6 + tried * 0.1,
                image_detail="auto",
            )
            response = actor_vlm_model.call_model(
                messages,
                decoding_args=action_vlm_decoding_args,
                return_list=False,
            ).strip()
            if validate_format(response):
                break
        if tried == 2:
            response = "[]"
        print(f"Image index {idx+1}: Processing image: {image_base_name}, true_objs: {true_objs_infor}, predicted_objs: {response}")
        answer.append([true_objs_infor, response, image_base_name])
    csv_save_folder = os.path.join(CURRENT_FOLDER, args.result_csv_folder, args.refine_type)
    if not os.path.exists(csv_save_folder):
        os.makedirs(csv_save_folder)
    csv_save_path = os.path.join(CURRENT_FOLDER, "predicted-objects-categories-list.csv")
    write_data_to_csv(csv_save_path, answer)
    total_money = get_total_money()
    print(f"The answer has been saved to predicted-objects-categories-list.csv in {csv_save_folder}")
    print(f"Total money: {total_money}")
    print("Done!")
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate object candidates for each image using VLM")
    parser.add_argument("--result_csv_folder", default="GPT4-V-result", type=str)
    parser.add_argument("--vlm_model", type=str, default="gpt-4-vision-preview", help="The VLM model to use")
    parser.add_argument("--refine_type", default="no-refine", type=str)
    parser.add_argument("--image_obj_infor_path", default="no-refine-image-obj.csv", type=str)
    args = parser.parse_args()
    main(args)
    