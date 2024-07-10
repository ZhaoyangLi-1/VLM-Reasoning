import argparse
import os
import json
import sys
import requests
from base64 import b64decode
from pickle import loads
from collections import deque
import time
from utils import *
from utils.logger import Global_Logger, Task_Logger
import ast
import csv
from agi.utils.chatbot_utils import DecodingArguments, ChatBot
import re
import pandas as pd
import random
import copy
from collections import deque

ALFWORLD_DATA = os.getenv("ALFWORLD_DATA")
ALFWORLD_SAVE = os.getenv("ALFWORLD_SAVE")
CURRENT_FOLDER = os.path.dirname(os.path.abspath(__file__))
PARENT_FOLDER = os.path.dirname(CURRENT_FOLDER)
TASK_ENV_LIST_PATH = os.path.join(CURRENT_FOLDER, "simple_pick_up_task_json.txt")


def write_data_to_csv(filename, data):
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        for row in data:
            writer.writerow(row)
            print(f"Added row: {row}")


def format_action_spaces(big_objects):
    admissible_commands = ["go to " + element for element in big_objects]
    admissible_commands = [
        re.sub(r"sof(\d)", r"sofa \1", command) for command in admissible_commands
    ]
    return admissible_commands


def get_task_paths_from_txt():
    task_paths = []
    with open(TASK_ENV_LIST_PATH, "r") as f:
        for line in f:
            path = os.path.join(ALFWORLD_DATA, line.strip())
            task_paths.append(path)
    return sorted(task_paths)


def check_executable_go_to_actions(args, env_url):
    task_envs_list = get_task_paths_from_txt()
    data = []
    data.append(
        [
            "task_id",
            "task_path",
            "task_desc",
            "executable_go_to_actions",
            "no_executable_go_to_actions",
        ]
    )
    for task_idx, task_env in enumerate(task_envs_list):
        text = b64decode(
            eval(requests.post(env_url + "/reset", json={"json_file": task_env}).text)
        )
        obs, infos = loads(text)
        receps, task_desc, ini_obs = format_initial_obs_to_get_rceps(obs)
        actions = format_action_spaces(receps)
        executable_go_to_actions = []
        no_executable_go_to_actions = []
        for action in actions:
            text = b64decode(
                eval(requests.post(env_url + "/step", json={"action": action}).text)
            )
            obs, _, _, _ = loads(text)
            obs = obs[0]
            # print(obs)
            if "Nothing happens." in obs:
                no_executable_go_to_actions.append(action)
            else:
                executable_go_to_actions.append(action)
        task_path = os.path.relpath(task_env, start=ALFWORLD_DATA)
        print(
            f"task_idx: {task_idx}, task_path: {task_path}, task_desc: {task_desc}, executable_go_to_actions: {executable_go_to_actions}, no_executable_go_to_actions: {no_executable_go_to_actions}"
        )
        data.append(
            [
                task_idx,
                task_env,
                task_desc,
                executable_go_to_actions,
                no_executable_go_to_actions,
            ]
        )
    csv_save_path = os.path.join(CURRENT_FOLDER, args.csv_save_name + ".csv")
    write_data_to_csv(csv_save_path, data)
    print(f"Data saved to {csv_save_path}")


def generate_balanced_go_to_actions(
    executable_go_to_actions, no_executable_go_to_actions, all_executable_go_to_actions
):
    action_executable_dict = {}

    for action in executable_go_to_actions:
        action_executable_dict[action] = 1

    non_executable_candidates = list(
        (all_executable_go_to_actions | set(no_executable_go_to_actions))
        - set(executable_go_to_actions)
    )
    sample_size = len(executable_go_to_actions)
    non_executable_sample = random.sample(non_executable_candidates, sample_size)
    for action in non_executable_sample:
        action_executable_dict[action] = 0
    return action_executable_dict


def get_prompts():
    action_prompt_path = os.path.join(
        CURRENT_FOLDER, "prompts/alfworld/test-ground/llm/action_prompt.txt"
    )
    with open(action_prompt_path, "r") as f:
        action_prompt = f.read().strip()
    return action_prompt

def extract_object_place(task_desk):
    match = re.search(r"\b(\w+)\s+(in|on)\s+(\w+)", task_desk)
    if match:
        return (match.group(1), match.group(3))
    return None


def check_pick_can_be_done(admissble_actions, task_object):
    task_object = f"take {task_object} "
    for admissble_action in admissble_actions:
        if task_object in admissble_action:
            return True
    return False


def map_reponse_from_vlm(response):
    try:
        data = json.loads(response)
        action = data.get("action", "No action")
        thought = data.get("thought", "No thought")
    except json.JSONDecodeError:
        thought, action = "No thought", "No action"
    
    return thought, action


def formorlize_action(executable_go_to_actions, target_object, tried_actions):
    pick_action = f"take {target_object}"
    if len(executable_go_to_actions) == 0:
        go_to_action = tried_actions[0]
    else:
        go_to_action = executable_go_to_actions[0]

    return f"[{go_to_action}, {pick_action}]"


def check_grounding_infor_in_pick(args, env_url):
    action_basic_prompt = get_prompts()
    action_vlm_decoding_args = DecodingArguments(
        max_tokens=8192,
        n=1,
        temperature=0.7,
        image_detail="auto",
    )
    actor_vlm_model = ChatBot(args.lm_model)
    print(f"VLM Model: {args.lm_model}")

    csv_save_path = os.path.join(CURRENT_FOLDER, args.csv_save_name + ".csv")
    df = pd.read_csv(csv_save_path)
    data = []
    
    if os.path.exists(os.path.join(CURRENT_FOLDER, "wrong_task.csv")):
        wrong_task_csv = pd.read_csv(os.path.join(CURRENT_FOLDER, "wrong_task.csv"))
    wrong_task_ids = list(wrong_task_csv["task_id"].values)
    # data.append(['task_id', 'task_path', 'task_desc', 'image_save_path', 'prompt', 'posterior_action', 'successor_action', 'thought', 'is_pick_can_be_done', 'vlm_succeed', 'target_object', 'place', 'admissible_commands_env'])
    columns = [
        "task_id",
        "task_path",
        "task_desc",
        "image_save_path",
        "prompt",
        "posterior_action",
        "successor_action",
        "thought",
        "is_pick_can_be_done",
        "vlm_succeed",
        "target_object",
        "place",
        "grounded_objects",
        "admissible_commands_env",
    ]
    image_save_folder = os.path.join(CURRENT_FOLDER, "test-grounded-images")
    if not os.path.exists(image_save_folder):
        os.makedirs(image_save_folder)
    for _, row in df.iterrows():
        task_id = row["task_id"]
        
        if task_id in wrong_task_ids:
            continue
        
        task_path = row["task_path"]
        task_desc = row["task_desc"]
        executable_go_to_actions = eval(row["executable_go_to_actions"])
        requests.post(
            env_url + "/reset",
            json={"json_file": os.path.join(ALFWORLD_DATA, task_path)},
        )

        target_object, place = extract_object_place(task_desc)

        task_id_image_save_folder = os.path.join(image_save_folder, f"task-{task_id}")
        if not os.path.exists(task_id_image_save_folder):
            os.makedirs(task_id_image_save_folder)
        tried_actions = deque()
        while executable_go_to_actions:
            posterior_action = executable_go_to_actions.pop(0)
            tried_actions.append(posterior_action)
            text = b64decode(
                eval(
                    requests.post(
                        env_url + "/step", json={"action": posterior_action}
                    ).text
                )
            )
            obs, _, done, infos = loads(text)
            admissible_commands = infos["admissible_commands"][0]
            obs = obs[0]

            count_dic, _, image, _ = get_obj_infor(env_url)
            file_name = copy.deepcopy(posterior_action).replace(" ", "-")
            image_save_path = os.path.join(
                task_id_image_save_folder, f"{file_name}.png"
            )
            image.save(image_save_path, "JPEG")
            grounded_objects = [
                object_name.lower() for object_name in list(count_dic.keys())
            ]
            print(f"Grounded Objects: {grounded_objects}")
            is_pick_can_be_done = check_pick_can_be_done(
                admissible_commands, target_object
            )

            
            admissible_commands_prompt = formorlize_action(
                executable_go_to_actions, target_object, tried_actions
            )
            prompt = action_basic_prompt.format(
                object_list=str(grounded_objects),
                task_desc=task_desc,
                admissible_commands=admissible_commands_prompt,
                task_related_object=target_object,
            )
            messages = {"text": prompt}

            response = actor_vlm_model.call_model(
                messages, decoding_args=action_vlm_decoding_args, return_list=False
            ).strip()
            thought, successor_action = map_reponse_from_vlm(response)

            if is_pick_can_be_done and "take" in successor_action:
                action_succeed = True
            elif "go to" in successor_action and not is_pick_can_be_done:
                action_succeed = True
            else:
                action_succeed = False
            print(
                f"Posterior Action : {posterior_action},  Successor Action: {successor_action},  Thought: {thought}, Action Succeed: {action_succeed}"
            )

            # data.append([task_id, task_path, task_desc, image_save_path, prompt, posterior_action, successor_action, thought, is_pick_can_be_done, action_succeed, target_object, place, grounded_objects, admissible_commands])
            data.append(
                {
                    "task_id": task_id,
                    "task_path": task_path,
                    "task_desc": task_desc,
                    "image_save_path": image_save_path,
                    "prompt": prompt,
                    "posterior_action": posterior_action,
                    "successor_action": successor_action,
                    "thought": thought,
                    "is_pick_can_be_done": is_pick_can_be_done,
                    "vlm_succeed": action_succeed,
                    "target_object": target_object,
                    "place": place,
                    "grounded_objects": grounded_objects,
                    "admissible_commands_env": admissible_commands,
                }
            )
    result_df = pd.DataFrame(data, columns=columns)
    result_csv_save_path = os.path.join(CURRENT_FOLDER, "grounded_pick_planning.csv")
    result_df.to_csv(result_csv_save_path, index=False)
    print(f"Data saved to {result_csv_save_path}")


def get_take_action(admissible_commands, target_object):
    for admissible_command in admissible_commands:
        take_action = f"take {target_object}"
        if take_action in admissible_command:
            if target_object in admissible_command:
                return admissible_command


def get_go_to_place_action(admissible_commands, place):
    for admissible_command in admissible_commands:
        go_to_place = f"go to {place}"
        if go_to_place in admissible_command:
            return admissible_command

def get_object_list_prompt():
    object_list_prompt_path = os.path.join(
        CURRENT_FOLDER, "prompts/alfworld/use-memory/list_objects.txt"
    )
    with open(object_list_prompt_path, "r") as f:
        object_list_prompt = f.read().strip()
    return object_list_prompt

def check_put_can_be_done(args, env_url):
    object_basic_list_prompt = get_object_list_prompt()
    action_vlm_decoding_args = DecodingArguments(
        max_tokens=8192,
        n=1,
        temperature=0.7,
        image_detail="auto",
    )
    actor_vlm_model = ChatBot(args.lm_model)
    print(f"VLM Model: {args.lm_model}")

    put_prompt_path = os.path.join(
        CURRENT_FOLDER, "prompts/alfworld/test-ground/llm/put_object.txt"
    )
    with open(put_prompt_path, "r") as f:
        put_basic_prompt = f.read().strip()
        assert put_basic_prompt is not None
    
    csv_save_path_pick = os.path.join(CURRENT_FOLDER, "grounded_pick_planning.csv")
    df_pick = pd.read_csv(csv_save_path_pick)
    df_task_group = df_pick.groupby("task_id")

    executable_and_not_for_30_tasks_path = os.path.join(
        CURRENT_FOLDER, args.csv_save_name + ".csv"
    )
    df_executable_for_all_tasks = pd.read_csv(executable_and_not_for_30_tasks_path)
    
    if os.path.exists(os.path.join(CURRENT_FOLDER, "wrong_task.csv")):
        wrong_task_csv = pd.read_csv(os.path.join(CURRENT_FOLDER, "wrong_task.csv"))
    wrong_task_ids = list(wrong_task_csv["task_id"].values)

    wrong_task_data = []
    data = []
    columns = [
        "task_id",
        "task_path",
        "task_desc",
        "image_save_path",
        "prompt",
        "posterior_action",
        "go_to_action",
        "successor_action",
        "thought",
        "is_vlm_response_pu",
        "action_succeed",
        "target_object",
        "place",
        "admissible_commands_env",
    ]

    for task_id, group in df_task_group:
        # if task_id != 5:
        #     continue
        # if task_id == 5:
        #     breakpoint()
        if task_id in wrong_task_ids:
            continue
        executable_for_all_the_task = ast.literal_eval(
            df_executable_for_all_tasks[
                df_executable_for_all_tasks["task_id"] == task_id
            ]["executable_go_to_actions"].values[0]
        )
        id_pick_can_be_done_row = group[group["is_pick_can_be_done"] == True]
        if id_pick_can_be_done_row.empty:
            wrong_task_data.append([task_id, group["task_path"].values[0], group["task_desc"].values[0]], "No pick can be done")
        place = id_pick_can_be_done_row["place"].values[0]
        target_object = id_pick_can_be_done_row["target_object"].values[0]
        posterior_action = id_pick_can_be_done_row["posterior_action"].values[0]
        admissible_commands_env = ast.literal_eval(
            id_pick_can_be_done_row["admissible_commands_env"].values[0]
        )
        pick_action = get_take_action(admissible_commands_env, target_object)
        task_desc = id_pick_can_be_done_row["task_desc"].values[0]
        task_path = id_pick_can_be_done_row["task_path"].values[0]

        requests.post(
            env_url + "/reset",
            json={"json_file": os.path.join(ALFWORLD_DATA, task_path)},
        )

        text = b64decode(
            eval(
                requests.post(env_url + "/step", json={"action": posterior_action}).text
            )
        )
        obs, _, done, infos = loads(text)
        admissible_commands_env = infos["admissible_commands"][0]
        obs = obs[0]
        assert "Nothing happens." not in obs

        text = b64decode(
            eval(requests.post(env_url + "/step", json={"action": pick_action}).text)
        )

        obs, _, done, infos = loads(text)
        admissible_commands_env = infos["admissible_commands"][0]
        obs = obs[0]
        assert "Nothing happens." not in obs

        # is_go_to_put_place_done = False
        # tried_go_to_actions = deque()
        # tried = 0
        go_to_place = get_go_to_place_action(executable_for_all_the_task, place)
        assert go_to_place is not None
        text = b64decode(
            eval(requests.post(env_url + "/step", json={"action": go_to_place}).text)
        )
        obs, _, done, infos = loads(text)
        admissible_commands_env = infos["admissible_commands"][0]
        obs = obs[0]
        assert "Nothing happens." not in obs
       
        for admissible_command in admissible_commands_env:
            if "put" in admissible_command:
                target_object_action = f"put {target_object}"
                if target_object_action in admissible_command:
                    put_action = admissible_command
                    break
        filtered_actions = [
            action for action in executable_for_all_the_task if action != go_to_place
        ]
        go_to_action = random.choice(filtered_actions)

        admissible_commands = f"[{go_to_action}, {put_action}]"

        count_dic, _, image, _ = get_obj_infor(env_url)
        image_save_basic_folder = os.path.join(CURRENT_FOLDER, "put-ground-images")
        image_task_save_folder = os.path.join(
            image_save_basic_folder, f"task-{task_id}"
        )
        if not os.path.exists(image_task_save_folder):
            os.makedirs(image_task_save_folder)
        image_save_path = os.path.join(image_task_save_folder, f"put-image.png")
        image.save(image_save_path, "JPEG")
        
        grounded_objects = [
            object_name.lower() for object_name in list(count_dic.keys())
        ]
        
        if len(grounded_objects) == 0:
            messages = {"image": [image], "text": object_basic_list_prompt}
            grounded_objects = actor_vlm_model.call_model(
                messages, decoding_args=action_vlm_decoding_args, return_list=False
            ).strip()
        else:
            grounded_objects = str(grounded_objects)   
        
        put_prompt = put_basic_prompt.format(
            object_list=grounded_objects,
            admissible_commands=admissible_commands,
            task_desc=task_desc,
            task_related_object=target_object,
            place=place,
        )
        messages = {"text": put_prompt}
        response = actor_vlm_model.call_model(
            messages, decoding_args=action_vlm_decoding_args, return_list=False
        ).strip()
        thought, successor_action = map_reponse_from_vlm(response)
        if "put" in successor_action:
            is_vlm_response_pu = True
        else:
            is_vlm_response_pu = False
        text = b64decode(
            eval(
                requests.post(env_url + "/step", json={"action": successor_action}).text
            )
        )
        obs, _, done, infos = loads(text)
        admissible_commands_env = infos["admissible_commands"][0]
        obs = obs[0]
        if "Nothing happens." in obs:
            action_succeed = False
        else:
            action_succeed = True

        data.append(
            {
                "task_id": task_id,
                "task_path": task_path,
                "task_desc": task_desc,
                "image_save_path": image_save_path,
                "prompt": put_prompt,
                "posterior_action": posterior_action,
                "go_to_action": go_to_place,
                "successor_action": successor_action,
                "thought": thought,
                "is_vlm_response_pu": is_vlm_response_pu,
                "action_succeed": action_succeed,
                "target_object": target_object,
                "place": place,
                "admissible_commands_env": admissible_commands_env,
            }
        )
        print(
            f"Task ID: {task_id}, Task Path: {task_path}, Task Desc: {task_desc}, Image Save Path: {image_save_path}, Prompt: {put_prompt}, Posterior Action: {posterior_action}, Go To Action: {go_to_place}, Successor Action: {successor_action}, Thought: {thought}, Is VLM Response Pu: {is_vlm_response_pu}, Action Succeed: {action_succeed}, Target Object: {target_object}, Place: {place}, Admissible Commands Env: {admissible_commands}"
        )

    result_df = pd.DataFrame(data, columns=columns)
    result_csv_save_path = os.path.join(CURRENT_FOLDER, "grounded_put_planning.csv")
    result_df.to_csv(result_csv_save_path, index=False)
    print(f"Data saved to {result_csv_save_path}")


def main(args):
    env_url = "http://127.0.0.1:" + str(args.env_url)
    set_dic = {"env_type": "visual", "batch_size": 1}
    requests.post(env_url + "/set_environment", json=set_dic).text
    if args.check_executable:
        check_executable_go_to_actions(args, env_url)
    if args.check_pick:
        check_grounding_infor_in_pick(args, env_url)
    if args.check_put:
        check_put_can_be_done(args, env_url)
    requests.post(env_url + "/close", json={})
    print("Closed the environment")
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lm_model", type=str, default="gpt-4-turbo-2024-04-09")
    parser.add_argument(
        "--csv_save_name", default="executable-and-not-for-30-tasks", type=str
    )
    parser.add_argument("--env_url", type=str, default=3000)
    parser.add_argument("--check_executable", action="store_true")
    parser.add_argument("--check_pick", action="store_true")
    parser.add_argument("--check_put", action="store_true")
    args = parser.parse_args()
    main(args)
