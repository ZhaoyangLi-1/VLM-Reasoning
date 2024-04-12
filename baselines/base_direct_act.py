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

os.environ["AGI_ROOT"] = "/home/zhaoyang/projects/neural-reasoning"
sys.path.append(os.path.join(os.environ["AGI_ROOT"]))
from agi.utils.openai_utils import get_total_money
from agi.utils.chatbot_utils import DecodingArguments, ChatBot


ALFWORLD_DATA = os.getenv("ALFWORLD_DATA")
ALFWORLD_SAVE = os.getenv("ALFWORLD_SAVE")
CURRENT_FOLDER = os.path.dirname(os.path.abspath(__file__))
PARENT_FOLDER = os.path.dirname(CURRENT_FOLDER)
PROMPT_PATH = os.path.join(PARENT_FOLDER, "prompts/alfworld-prompts")


def generate_action_prompt(task_desc, admissible_commands, task_hint, num_images, ini_obs):
    images = "".join([f"Image{i}:\n<image>\n" for i in range(1, num_images + 1)])
    return images + basic_action_prompt.format(
        ini_obs=ini_obs,
        task_desc=task_desc,
        admissible_commands=admissible_commands,
        task_hint=task_hint,
    )


def get_prompts(args):
    
    def get_task_hints(task_hints_prompt_path):
        task_hints = {}
        for filepath, _, filenames in os.walk(task_hints_prompt_path):
            for filename in filenames:
                take_name = filename.split(".")[0]
                with open(os.path.join(filepath, filename), "r") as f:
                    task_hints[take_name] = f.read().replace("\n", " ")
        return task_hints
    
    baseline_action_selector_prompt_path = os.path.join(
        PROMPT_PATH, "baselines", args.prompt_path
    )
    task_hints_prompt_path = os.path.join(PROMPT_PATH, "task-hints")

    with open(baseline_action_selector_prompt_path, "r") as f:
        basic_action_prompt = f.read()

    task_hints = get_task_hints(task_hints_prompt_path)

    return basic_action_prompt, task_hints


def delete_inefficient_action(admissible_commands, no_try_actions):
    return [element for element in admissible_commands if element not in no_try_actions]


def test_tasks(args):
    print(f"The max number of images: {args.max_images}")
    global basic_action_prompt, task_hints_prompt
    basic_action_prompt, task_hints = get_prompts(args)
    env_url = "http://127.0.0.1:" + str(args.env_url)
    # initial VLM and LLM model
    print(f"VLM Model: {args.vlm_model}")
    # Setup VLM(gpt4-v) model as action selector and current attempts relfector
    action_vlm_decoding_args = DecodingArguments(
        max_tokens=8192,
        n=1,
        temperature=0.7,
        image_detail="auto",
    )
    actor_vlm_model = ChatBot(args.vlm_model)
    # recored the number of success of tasks
    num_succeess = 0
    # Check whether continue from last time
    save_path = os.path.join(ALFWORLD_SAVE, args.save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    total_tasks_log = Global_Logger(
        os.path.join(save_path, f"succeed-num-begin-{args.begin_task}.log")
    )

    json_file_list = total_tasks_log.skip_succeed(PARENT_FOLDER, args)
    set_dic = {"env_type": "visual", "batch_size": 1}
    requests.post(env_url + "/set_environment", json=set_dic).text
    for task_idx, json_file in enumerate(json_file_list):
        # Get task type
        with open(json_file, "r") as f_task:
            task_json = json.load(f_task)
            task_type = task_json["task_type"]
            f_task.close()
        task_hint = task_hints[task_type]
        succeed = False
        task_idx = task_idx + total_tasks_log.num_done + args.begin_task
        print(f"Current task: {task_idx}")
        task_save_root = os.path.join(save_path, f"task-{task_idx}")
        if not os.path.exists(task_save_root):
            os.makedirs(task_save_root)
        task_log_save_path = os.path.join(task_save_root, f"task-{task_idx}.log")
        task_logger = Task_Logger(task_log_save_path, task_idx)
        text = b64decode(
            eval(requests.post(env_url + "/reset", json={"json_file": json_file}).text)
        )
        obs, infos = loads(text)
        admissible_commands = format_admissible_commands(
            infos["admissible_commands"][0]
        )
        # first one is initial observation
        ini_obs, task_desc = format_obs_task_desc(obs)
        ini_obs = refine_ini_obs(ini_obs)
        # Set image_queue
        images_queue = deque(maxlen=args.max_images)
        image_paths_queue = deque(maxlen=args.max_images)
        no_try_actions = deque(maxlen=3)
        images_log = []
        for step in range(args.max_step):
            image_root = os.path.join(task_save_root, f"images")
            if not os.path.exists(image_root):
                os.makedirs(image_root)
            image_path = os.path.join(image_root, f"task-{task_idx}-{step}.jpg")
            image = get_image(env_url, args.is_ins_seg)
            image.save(image_path, "JPEG")
            images_queue.append(image)
            images_log.append(image)
            image_paths_queue.append(image_path)
            prompt = generate_action_prompt(
                task_desc, admissible_commands, task_hint, len(images_queue), ini_obs
            )
            messages = {
                "text": prompt,
                "images": list(images_queue),
            }
            start_time = time.time()
            response = actor_vlm_model.call_model(
                messages,
                decoding_args=action_vlm_decoding_args,
                return_list=False,
            ).strip()
            end_time = time.time()
            action = refine_action(response).strip()
            no_try_actions.append(action)
            text = b64decode(
                eval(requests.post(env_url + "/step", json={"action": action}).text)
            )
            obs, _, done, infos = loads(text)
            obs, _, done, admissible_commands = (
                obs[0],
                infos["won"][0],
                done[0],
                infos["admissible_commands"][0],
            )
            text_log = task_logger.wirte_and_get_task_log(
                task_idx,
                step,
                prompt,
                response,
                action,
                start_time,
                end_time,
                get_total_money(),
                obs,
            )
            admissible_commands = delete_inefficient_action(admissible_commands, no_try_actions)
            admissible_commands, _ = format_admissible_commands(
                admissible_commands
            )
            succeed = done
            goal_condition_success_rate = infos["goal_condition_success_rate"][0]
            html_path = os.path.join(task_save_root, "html-files")
            if not os.path.exists(html_path):
                os.makedirs(html_path)
            if succeed:
                num_succeess += 1
                image_path = os.path.join(
                    image_root, f"task-{task_idx}-step-{step+1}.jpg"
                )
                image = get_image(env_url, args.is_ins_seg)
                image.save(image_path, "JPEG")
                images_queue.append(image)
                images_log.append(image)
                image_paths_queue.append(image_path)
                # images_dic_list.append({"images": image, "path": image_path})
                total_tasks_log.write(
                    f"{task_idx} Path:{json_file}: SUCCEED, Goal condition success rate: {goal_condition_success_rate}\n"
                )
                save_video(
                    images_log,
                    os.path.join(task_save_root, f"task-{task_idx}.mp4"),
                )
                text_log += "\nSUCCEED\n"
                generate_html_with_task_log(
                    text_log,
                    image_paths_queue,
                    os.path.join(html_path, f"task-{task_idx}-step-{step}.html"),
                )
                break
            text_log += "\nUNSUCCEED\n"
            generate_html_with_task_log(
                text_log,
                image_paths_queue,
                os.path.join(html_path, f"task-{task_idx}-step-{step}.html"),
            )
        if not succeed:
            save_video(
                    images_log,
                    os.path.join(task_save_root, f"task-{task_idx}.mp4"),
            )
            total_tasks_log.write(
                f"{task_idx} Path:{json_file}: UNSUCCEED, Goal condition success rate: {goal_condition_success_rate}\n"
            )
    total_money = get_total_money()
    total_tasks_log.write(f"\nTotal number: {len(json_file_list)}\n")
    total_tasks_log.write(f"Total succeed number: {num_succeess}\n")
    total_tasks_log.write(f"Total money Cost: {total_money}\n")

    requests.post(env_url + "/close", json={})
