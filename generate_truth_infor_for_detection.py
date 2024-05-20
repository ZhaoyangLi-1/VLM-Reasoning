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
from utils.alfworld_utils import get_obj_infor
import csv

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
    # receps = ["drawer", "dresser", "fridge", "microwave", "cabinet", "safe"]
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
        # text_obs_for_each_action = set()
        for action, big_object in zip(actions, big_objects):
            text = b64decode(
                eval(requests.post(ENV_URL + "/step", json={"action": action}).text)
            )
            obs, _, _, _ = loads(text)
            obs = obs[0]
            if "Nothing happens." in obs or "nothing" in obs:
                continue
            # if big_object.split(" ")[0] in receps and "closed" in obs:
            #     open_action = f"open {big_object}"
            #     text = b64decode(
            #         eval(requests.post(ENV_URL + "/step", json={"action": open_action}).text)
            #     )
            #     obs, _, _, _ = loads(text)
            #     obs = obs[0]
            #     print(f"Current action: {open_action}, obs: {obs}")
            #     if "Nothing happens." in obs:
            #         continue
            # obs = obs.split("you see ")
            # if len(obs) > 1:
            #     obs = obs[1]
            # else:
            #     continue
            # if obs in text_obs_for_each_action:
            #     continue
            # else:
            #     text_obs_for_each_action.add(obs)
            
            count_dict, instance_image, original_image, merged_obj_dic = get_obj_infor(ENV_URL)
            if len(count_dict) == 0 or len(merged_obj_dic) == 0:
                continue
            print(f"Current action: {action}, obs: {obs}, Image Information: {obs}, merged_obj_dic: {merged_obj_dic}, count_dict: {count_dict}")
            
            instance_image_save_folder = os.path.join(ALFWORLD_SAVE, f"images-new/{args.ins_image_save}")
            if os.path.exists(instance_image_save_folder) == False:
                os.makedirs(instance_image_save_folder)
            instance_image_save_path = os.path.join(instance_image_save_folder, f"{img_idx}.jpg")
            instance_image.save(instance_image_save_path, "JPEG")
            
            original_image_save_folder = os.path.join(ALFWORLD_SAVE, "images-new/original")
            if os.path.exists(original_image_save_folder) == False:
                os.makedirs(original_image_save_folder)
            original_image_save_path = os.path.join(original_image_save_folder, f"{img_idx}.jpg")
            if not os.path.exists(original_image_save_path):
                original_image.save(original_image_save_path, "JPEG")
            
            # if big_object.split(" ")[0] in receps and "closed" in obs:
            #     data.append([task_desc, f"{action} -> {open_action}", original_image_save_path, instance_image_save_path, count_dict, rel_task_path, merged_obj_dic])
            # else:
            data.append([task_desc, action, original_image_save_path, instance_image_save_path, count_dict, rel_task_path, merged_obj_dic])
            img_idx += 1
    csv_save_path = os.path.join(ALFWORLD_SAVE, args.csv_save_name + ".csv")
    write_data_to_csv(csv_save_path, data)
    requests.post(ENV_URL + "/close", json={})
    print(f"Data saved to {csv_save_path}")
    print("Done!")
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_save_name", default="edge-refine-new-image-obj", type=str)
    parser.add_argument("--ins_image_save", default="edge-refine-ins", type=str)
    args = parser.parse_args()
    main(args)
    
# You are in the middle of a room. Looking quickly around you, you see a bed 1, a sidetable 1, a drawer 1, a dresser 1, a drawer 2, a drawer 3, a drawer 4, a drawer 5, a drawer 6, a drawer 7, a drawer 8, a drawer 9, a drawer 10, a drawer 11, a safe 1, a laundryhamper 1, and a garbagecan 1.

