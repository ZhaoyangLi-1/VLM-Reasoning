import argparse
import numpy as np
import os
import json
import sys
import requests
from base64 import b64decode
from pickle import loads
from PIL import Image, ImageDraw, ImageFont
import random
import csv

os.environ["AGI_ROOT"] = "/home/zhaoyang/projects/neural-reasoning"
sys.path.append(os.path.join(os.environ["AGI_ROOT"]))
from agi.utils.chatbot_utils import DecodingArguments, ChatBot


ALFWORLD_DATA = os.getenv("ALFWORLD_DATA")
CURRENT_FOLDER = os.path.dirname(os.path.abspath(__file__))
ALFWORLD_SAVE = os.path.join(CURRENT_FOLDER, "test-detect")
PARENT_FOLDER = os.path.dirname(CURRENT_FOLDER)
PROMPT_PATH = os.path.join(PARENT_FOLDER, "prompts/alfworld-prompts")
TASK_ENV_LIST_PATH = os.path.join(CURRENT_FOLDER, "tasks/sub_exmaple_tasks_set.json")
ENV_URL = "http://127.0.0.1:3000"


def get_all_task_envs_list():
    task_envs_list = []
    with open(TASK_ENV_LIST_PATH, "rb") as f:
        relative_path_json = json.load(f)
        for _, rel_paths in relative_path_json.items():
            for rel_path in rel_paths:
                full_path = os.path.join(ALFWORLD_DATA, rel_path)
                task_envs_list.append(full_path)
    return sorted(task_envs_list)


def draw_instance_img(original_image, env_url, bbox_threshold=0, image_size_margin_ratio=0.005):
    def is_dark_color(color):
        r, g, b = color
        return (0.299 * r + 0.587 * g + 0.114 * b) < 128

    def is_inside(inner_bbox, outer_bbox):
        return (
            outer_bbox[0] <= inner_bbox[0] <= outer_bbox[2]
            and outer_bbox[0] <= inner_bbox[2] <= outer_bbox[2]
            and outer_bbox[1] <= inner_bbox[1] <= outer_bbox[3]
            and outer_bbox[1] <= inner_bbox[3] <= outer_bbox[3]
        )

    def count_inner_boxes(obj_dic):
        result = {}
        for obj_id, bbox in obj_dic.items():
            num_inside = 0
            for other_id, other_bbox in obj_dic.items():
                if obj_id != other_id and is_inside(other_bbox, bbox):
                    num_inside += 1
            result[obj_id] = (bbox, num_inside)
        return result

    def get_average_color(image, bbox):
        x1, y1, x2, y2 = bbox
        area = image.crop((x1, y1, x2, y2))
        avg_color = np.array(area).mean(axis=(0, 1))
        return avg_color

    def is_overlapping(new_pos, existing_positions):
        x1_new, y1_new, x2_new, y2_new = new_pos
        for pos in existing_positions:
            x1, y1, x2, y2 = pos
            if not (x2_new < x1 or x2 < x1_new or y2_new < y1 or y2 < y1_new):
                return True
        return False
    
    def random_color():
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    objects_receps = loads(
        b64decode(eval(requests.post(env_url + "/get_objects_receps", json={}).text))
    )[0]
    instance_segs_list, instance_detections2D_list = loads(
        b64decode(
            eval(requests.post(env_url + "/get_instance_seg_and_id", json={}).text)
        )
    )

    _, instance_detections2D = (
        instance_segs_list[0],
        instance_detections2D_list[0],
    )

    draw = ImageDraw.Draw(original_image)
    font = ImageFont.truetype("/home/zhaoyang/.fonts/Helvetica.ttf", size=50)  # ImageFont.load_default() #ImageFont.truetype("/home/zhaoyang/.fonts/Helvetica.ttf", size=50) 

    drawn_text_positions = []
    obj_dic = {}
    for obj_id, obj in objects_receps.items():
        if obj_id in instance_detections2D:
            bbox = instance_detections2D[obj_id].tolist()
        else:
            continue
        text = str(obj["num_id"])
        obj_dic[text] = bbox

    obj_dic = count_inner_boxes(obj_dic)
    merged_obj_dic = {}
    # Second pass to draw text and overlay
    img_width, img_height = original_image.size
    for obj_text_id, (bbox, num_inside) in obj_dic.items():
        x1, y1, x2, y2 = bbox
        
        margin_width = image_size_margin_ratio * img_width
        margin_height = image_size_margin_ratio * img_height
        
        in_x1 = max(x1, 0)
        in_y1 = max(y1, 0)
        in_x2 = min(x2, img_width)
        in_y2 = min(y2, img_height)
        effective_width = max(0, in_x2 - in_x1)
        effective_height = max(0, in_y2 - in_y1)
        effective_area = effective_width * effective_height
        
        # Calculate the total area of the bounding box
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        bbox_area = bbox_width * bbox_height
        
        image_area = img_width * img_height
        bbox_percentage_of_image = bbox_area / image_area
        
        if (x1 <= margin_width or y1 <= margin_height or x2 >= img_width - margin_width or y2 >= img_height - margin_height) and bbox_percentage_of_image < 0.2:
            continue
        
        color = random_color()
        draw.rectangle(bbox, outline=color, width=2)
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width, text_height = (
            text_bbox[2] - text_bbox[0],
            text_bbox[3] - text_bbox[1],
        )
        # create object and bbox dictionary
        object_type = obj_text_id.split()[0]
        if object_type not in merged_obj_dic:
            merged_obj_dic[object_type] = []
        merged_obj_dic[object_type].append(bbox)

        # Center the text in the bounding box
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        text_x = center_x - text_width / 2 + 40
        text_y = center_y - text_height / 2

        new_pos = (text_x, text_y, text_x + text_width, text_y + text_height)
        while is_overlapping(new_pos, drawn_text_positions):
            text_x += 10
            text_y += 10
            new_pos = (text_x, text_y, text_x + text_width, text_y + text_height)

        avg_bg_color = get_average_color(original_image, new_pos)
        text_color = "white" if is_dark_color(avg_bg_color) else "black"
        draw.text((text_x, text_y), obj_text_id, fill=text_color)
        drawn_text_positions.append(new_pos)
        
    return original_image, merged_obj_dic


def get_image(env_url, is_ins_seg, bbox_threshold=0):
    text = b64decode(eval(requests.post(env_url + "/get_frames", json={}).text))
    image = loads(text).squeeze()[:, :, ::-1].astype("uint8")
    image = Image.fromarray(image)
    if is_ins_seg:
        image, merged_obj_dic = draw_instance_img(image, env_url, bbox_threshold)
        return image, merged_obj_dic
    return image


def format_obs(obs):
    obs = obs[0].replace("\n\n", "\n").split("\n")
    obs_desc = obs[1]
    task_start_start_pos = obs[2].find(":") + 1
    task_desc = obs[2][task_start_start_pos:].strip()
    big_objects = obs_desc.split("you see a ")[1].split(", ")
    big_objects[-1] = big_objects[-1].replace("and a ", "")
    big_objects[-1] = big_objects[-1].replace(".", "")
    big_objects = [s.replace("a ", "") for s in big_objects]
    return big_objects, task_desc, obs_desc

def format_action_spaces(big_objects):
    return ["go to " + element for element in big_objects]
    
    
def get_obj_infor(bbox_threshold):
    instance_image, merged_obj_dic = get_image(ENV_URL, True, bbox_threshold)
    original_image = get_image(ENV_URL, False)
    objects_receps = loads(
        b64decode(eval(requests.post(ENV_URL + "/get_objects_receps", json={}).text))
    )[0]
    instance_segs_list, instance_detections2D_list = loads(
        b64decode(
            eval(requests.post(ENV_URL + "/get_instance_seg_and_id", json={}).text)
        )
    )
    _, instance_detections2D = (
        instance_segs_list[0],
        instance_detections2D_list[0],
    )
    count_dict = {}
    for obj_id, obj in objects_receps.items():
        if obj_id in instance_detections2D:
            object_type = obj["object_type"]
            if object_type in count_dict:
                count_dict[object_type] += 1
            else:
                count_dict[object_type] = 1
        else:
            continue
        
    return count_dict, instance_image, original_image, merged_obj_dic

   
def dict_to_str(d):
    return json.dumps(d)


def write_data_to_csv(filename, data):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        for row in data:
            if type(row[-2]) == dict:
                row[-2] = dict_to_str(row[-2])
            writer.writerow(row)
            print(f'Added row: {row}')


def main(args):
    data = []
    receps = ["drawer", "dresser", "fridge", "microwave", "cabinet", "safe"]
    data.append(['task_desc', 'action', 'original_image_path', 'ins_image_path', 'object_infor', 'task_path', 'object_bboxes'])
    task_envs_list = get_all_task_envs_list()
    set_dic = {"env_type": "visual", "batch_size": 1}
    requests.post(ENV_URL + "/set_environment", json=set_dic).text
    img_idx = 0
    for idx, task_env in enumerate(task_envs_list):
        rel_task_path = os.path.relpath(task_env, ALFWORLD_DATA)
        text = b64decode(
            eval(requests.post(ENV_URL + "/reset", json={"json_file": task_env}).text)
        )
        obs, _ = loads(text)
        big_objects, task_desc, _ = format_obs(obs)
        print(f"Current task: {idx}, task_desc: {task_desc}")
        actions = format_action_spaces(big_objects)
        text_obs_for_each_action = set()
        for action, big_object in zip(actions, big_objects):
            text = b64decode(
                eval(requests.post(ENV_URL + "/step", json={"action": action}).text)
            )
            obs, _, _, _ = loads(text)
            obs = obs[0]
            print(f"Current action: {action}, obs: {obs}")
            if "Nothing happens." in obs:
                continue
            if big_object.split(" ")[0] in receps and "closed" in obs:
                open_action = f"open {big_object}"
                text = b64decode(
                    eval(requests.post(ENV_URL + "/step", json={"action": open_action}).text)
                )
                obs, _, _, _ = loads(text)
                obs = obs[0]
                print(f"Current action: {open_action}, obs: {obs}")
                if "Nothing happens." in obs:
                    continue
            obs = obs.split("you see ")
            if len(obs) > 1:
                obs = obs[1]
            else:
                continue
            if obs in text_obs_for_each_action:
                continue
            else:
                text_obs_for_each_action.add(obs)
            
            count_dict, instance_image, original_image, merged_obj_dic = get_obj_infor(args.bbox_threshold)
            print(f"Image Information: {obs}, merged_obj_dic: {merged_obj_dic}")
            
            instance_image_save_folder = os.path.join(ALFWORLD_SAVE, f"images-size-384/{args.ins_image_save}")
            if os.path.exists(instance_image_save_folder) == False:
                os.makedirs(instance_image_save_folder)
            instance_image_save_path = os.path.join(instance_image_save_folder, f"{img_idx}.jpg")
            instance_image.save(instance_image_save_path, "JPEG")
            
            original_image_save_folder = os.path.join(ALFWORLD_SAVE, "images-size-384/original")
            if os.path.exists(original_image_save_folder) == False:
                os.makedirs(original_image_save_folder)
            original_image_save_path = os.path.join(original_image_save_folder, f"{img_idx}.jpg")
            if not os.path.exists(original_image_save_path):
                original_image.save(original_image_save_path, "JPEG")
            
            if big_object.split(" ")[0] in receps and "closed" in obs:
                data.append([task_desc, f"{action} -> {open_action}", original_image_save_path, instance_image_save_path, count_dict, rel_task_path, merged_obj_dic])
            else:
                data.append([task_desc, action, original_image_save_path, instance_image_save_path, count_dict, rel_task_path, merged_obj_dic])
            img_idx += 1
    csv_save_path = os.path.join(ALFWORLD_SAVE, args.csv_save_name + ".csv")
    write_data_to_csv(csv_save_path, data)
    requests.post(ENV_URL + "/close", json={})
    print(f"Data saved to {csv_save_path}")
    print("Done!")
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_save_name", default="0-percent-refine-image-obj", type=str)
    parser.add_argument("--bbox_threshold", default=0, type=float)
    parser.add_argument("--ins_image_save", default="0-refine-ins", type=str)
    args = parser.parse_args()
    main(args)
    
# You are in the middle of a room. Looking quickly around you, you see a bed 1, a sidetable 1, a drawer 1, a dresser 1, a drawer 2, a drawer 3, a drawer 4, a drawer 5, a drawer 6, a drawer 7, a drawer 8, a drawer 9, a drawer 10, a drawer 11, a safe 1, a laundryhamper 1, and a garbagecan 1.

