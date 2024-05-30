import os
import sys
import json
import argparse
import csv
import re
from PIL import Image
from utils import read_data_from_csv
import ast
import random


os.environ["AGI_ROOT"] = "/home/zhaoyang/projects/neural-reasoning"
sys.path.append(os.path.join(os.environ["AGI_ROOT"]))
from agi.utils.chatbot_utils import DecodingArguments, ChatBot

CURRENT_FOLDER = os.path.dirname(os.path.abspath(__file__))

def camel_case_to_words(s):
    s = re.sub(r'(?<!^)(?=[A-Z])', ' ', s)
    return s.lower()

def extract_number(s):
    # Updated to handle non-integer strings gracefully
    match = re.search(r'\d+', s)
    return int(match.group()) if match else None


def get_all_objs(data_for_all_images):
    objects_for_all_images = set()
    for item in data_for_all_images:
        # item_dic =  ast.literal_eval(item['object_infor'])
        objects_for_all_images.update(item['object_infor'].keys())
    return list(objects_for_all_images)

def generate_balance_data(object_infor, all_objects):
    objects_in_image = list(object_infor.keys())
    num_objects_in_image = len(objects_in_image)
    available_objs_candidates = list(set(all_objects) - set(objects_in_image))
    selected_objs_candidates = random.sample(available_objs_candidates, min(len(available_objs_candidates), num_objects_in_image))
    final_objs_dict = object_infor.copy()
    for obj in selected_objs_candidates:
        final_objs_dict[obj] = 0 
    return final_objs_dict


def write_data_to_csv(filename, row):
    file_exists = os.path.isfile(filename) and os.path.getsize(filename) > 0
    with open(filename, 'a', newline='') as file:  # Change 'w' to 'a' to append to the file
        writer = csv.writer(file)
        
        if not file_exists:
            header = ['object_category', 'question_type', 'true_counts', 'predicted_counts', 'correctness', 'original_image_path']
            writer.writerow(header)
        
        writer.writerow(row)
        print(f'Added row: {row}')
        

def load_existing_data_and_find_max_image(filename):
    existing_data = set()
    max_image_base_name = None
    max_image_number = -1
    try:
        with open(filename, 'r', newline='') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                if len(row) > 5:
                    obj_name, image_base_name = row[0], row[5]
                    existing_data.add((obj_name, image_base_name))
                    image_number = int(re.search(r'(\d+)\.jpg', image_base_name).group(1))
                    if image_number > max_image_number:
                        max_image_number = image_number
                        max_image_base_name = image_base_name
    except FileNotFoundError:
        pass  
    return existing_data, max_image_base_name


def rewrite_csv_excluding_max_image(filename, existing_data, max_image_base_name):
    temp_data = []
    with open(filename, 'r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) > 5 and row[5] != max_image_base_name:
                temp_data.append(row)
    
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(temp_data)


def main(args):
    print(f"Start testing the model with {args.QA_Mode} question using {args.vlm_model}...")
    prompt_path = os.path.join(CURRENT_FOLDER, f"prompts/{args.QA_Mode}.txt")
    prompt_template = open(prompt_path, "r").read()

    action_vlm_decoding_args = DecodingArguments(
        max_tokens=2048,
        n=1,
        temperature=0.6,
        image_detail="auto",
    )
    actor_vlm_model = ChatBot(args.vlm_model)

    image_obj_infor_path = os.path.join(CURRENT_FOLDER, args.image_obj_infor_path)
    data = read_data_from_csv(image_obj_infor_path)

    csv_save_folder = os.path.join(CURRENT_FOLDER, args.result_csv_folder, args.refine_type)
    if not os.path.exists(csv_save_folder):
        os.makedirs(csv_save_folder)
    csv_save_path = os.path.join(csv_save_folder, f"{args.QA_Mode}-answer.csv")
    existing_data, max_image_base_name = load_existing_data_and_find_max_image(csv_save_path)
    
    if max_image_base_name:
        rewrite_csv_excluding_max_image(csv_save_path, existing_data, max_image_base_name)
        existing_data = {data for data in existing_data if data[1] != max_image_base_name}


    all_objs = get_all_objs(data)
    for row in data:
        original_image_pth = row["original_image_path"]
        image_base_name = os.path.basename(original_image_pth)
        image_number = int(re.search(r'(\d+)\.jpg', image_base_name).group(1))
        max_image_number = int(re.search(r'(\d+)\.jpg', max_image_base_name).group(1)) if max_image_base_name else -1

        if image_number < max_image_number:
            print(f"Skipping {image_base_name} as it's less than the max found.")
            continue  # Skip images with a number less than the max found
        
        image = Image.open(original_image_pth)
        object_infor = row["object_infor"]
        object_balanced_infor = generate_balance_data(object_infor, all_objs)
        
        for obj_name, counts in object_balanced_infor.items():
            if (obj_name, image_base_name) in existing_data and image_number != max_image_number:
                print(f"Skipping {obj_name} in {image_base_name} as it's already processed.")
                continue

            prompt = prompt_template.format(obj_name=camel_case_to_words(obj_name))
            # breakpoint()
            messages = {
                "text": prompt,
                "images": [image],
            }
            response = actor_vlm_model.call_model(
                messages,
                decoding_args=action_vlm_decoding_args,
                return_list=False,
            ).strip()
            
            if args.QA_Mode == "counting":
                while extract_number(response) is None:
                    response = actor_vlm_model.call_model(
                        messages,
                        decoding_args=action_vlm_decoding_args,
                        return_list=False,
                    ).strip()
                    
            counts_hat, correctness = process_response(response, counts, args.QA_Mode)
            write_data_to_csv(csv_save_path, [obj_name, args.QA_Mode, counts, counts_hat, correctness, image_base_name])

    print("Done!")


# Helper function to determine correctness and predicted count
def process_response(response, counts, QA_Mode):
    if "count" in QA_Mode:
        counts_hat = extract_number(response)
        correctness = "correct" if counts_hat == counts else "incorrect"
    else:
        if "Yes" in response and counts > 0 or "No" in response and counts == 0:
            correctness = "correct"
        else:
            correctness = "incorrect"
        counts_hat = 1 if "Yes" in response else 0
    return counts_hat, correctness

                    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_csv_folder", default="GPT4-V-result-latest", type=str)
    parser.add_argument("--vlm_model", default="gpt-4-vision-preview", type=str)
    parser.add_argument("--image_obj_infor_path", default="edge-refine-new-image-obj.csv", type=str)
    parser.add_argument("--QA_Mode", default="existence", type=str)
    parser.add_argument("--refine_type", default="edge-refine", type=str)
    args = parser.parse_args()
    main(args)
    
# python test_detect_with_VLM.py --result_csv_folder GPT4-V-result-balance --image_obj_infor_path edge-refine-image-obj-revise.csv --QA_Mode existence --refine_type edge-refine
# python test_detect_with_VLM.py --result_csv_folder GPT4-V-result-balance --image_obj_infor_path edge-refine-image-obj-revise.csv --QA_Mode counting --refine_type edge-refine
# python test_detect_with_VLM.py --result_csv_folder GPT4-V-result-latest --image_obj_infor_path edge-refine-new-image-obj.csv --QA_Mode existence --refine_type edge-refine
# python test_detect_with_VLM.py --result_csv_folder GPT4-V-result-latest --image_obj_infor_path edge-refine-new-image-obj.csv --QA_Mode counting --refine_type edge-refine

# If there is no {obj_name}, answer 0.