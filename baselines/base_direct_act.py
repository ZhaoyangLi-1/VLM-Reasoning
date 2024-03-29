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

os.environ["AGI_ROOT"] = "/home/zhaoyang/projects/neural-reasoning"
sys.path.append(os.path.join(os.environ["AGI_ROOT"]))
from agi.utils.openai_utils import get_total_money
from agi.utils.chatbot_utils import DecodingArguments, ChatBot


ALFWORLD_DATA = os.getenv("ALFWORLD_DATA")
ALFWORLD_SAVE = os.getenv("ALFWORLD_SAVE")
CURRENT_FOLDER = os.path.dirname(os.path.abspath(__file__))
PARENT_FOLDER = os.path.dirname(CURRENT_FOLDER)
PROMPT_PATH = os.path.join(PARENT_FOLDER, "prompts/alfworld-prompts")


# Generate action selection prompt
def generate_action_prompt(task_desc, admissible_commands, task_hint, num_images):
    return "<image>\n" * num_images + basic_action_prompt.format(
        task_desc=task_desc,
        admissible_commands=admissible_commands,
        task_hint=task_hint,
    )

def get_prompts():
    baseline_action_selector_prompt_path = os.path.join(PROMPT_PATH, "baselines/direct_choose_action.text")
    task_hints_prompt_path = os.path.join(PROMPT_PATH, "task-hints")

    with open(baseline_action_selector_prompt_path, "r") as f:
        basic_action_prompt = f.read()
    
    task_hints = get_task_hints(task_hints_prompt_path)
    
    return basic_action_prompt , task_hints
    

def test_scenes(args):
    print(f"The max number of images: {args.max_images}")
    global basic_action_prompt, task_hints_prompt
    basic_action_prompt, task_hints = get_prompts()
    env_url = "http://127.0.0.1:" + str(args.env_url)
    # initial VLM and LLM model
    print(f"VLM Model: {args.vlm_model}")
    # Setup VLM(gpt4-v) model as action selector and current attempts relfector
    actor_vlm_model = ChatBot("gpt-4-vision-preview")

    # recored the number of success of scenes
    num_succeess = 0

    # Check whether continue from last time
    save_path = os.path.join(ALFWORLD_SAVE, args.save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    succeed_num_save_path = os.path.join(
        save_path, f"succeed_num_begin_{args.begin_scene}.log"
    )
    done_json_paths = []
    no_head = False
    if os.path.exists(succeed_num_save_path):
        print("Continue from the last time")
        with open(succeed_num_save_path, "r") as f:
            lines = f.readlines()
            done_json_paths = get_done_paths(lines)
            for idx, line in enumerate(lines):
                if "SUCCEED" in line and "UNSUCCEED" not in line:
                    num_succeess += 1
                if idx == 0 and "Begin Scene:" not in line:
                    no_head = True
    else:
        no_head = True

    task_list_path = os.path.join(PARENT_FOLDER, args.task_list_path)
    json_file_list = get_path_tasks(task_list_path)

    scenes_steps = args.total_scene // args.num_server
    json_file_list = sorted(
        json_file_list[
            args.begin_scene : min(args.begin_scene + scenes_steps, args.total_scene)
        ]
    )
    json_file_list = sorted(
        [item for item in json_file_list if item not in done_json_paths]
    )
    num_done = len(done_json_paths)

    if no_head:
        with open(succeed_num_save_path, "a") as succeed_num_f:
            succeed_num_f.write(
                f"Begin Scene: {args.begin_scene}     End Scene: {args.begin_scene+scenes_steps-1}      Number of Server: {args.num_server}\n"
            )
            succeed_num_f.close()

    # Set envrionment
    set_dic = {"env_type": "visual", "batch_size": 1}
    requests.post(env_url + "/set_environment", json=set_dic).text
    for scene_idx, json_file in enumerate(json_file_list):
        # Get task type
        with open(json_file, "r") as f_task:
            task_json = json.load(f_task)
            task_type = task_json["task_type"]
        task_hint = task_hints[task_type]
        max_step = 50
        succeed = False

        # Setup Log path
        scene_idx = scene_idx + num_done + args.begin_scene
        print(f"Current scene: {scene_idx}")
        scene_save_root = os.path.join(save_path, f"scene_{scene_idx}")
        if not os.path.exists(scene_save_root):
            os.makedirs(scene_save_root)
        scene_save_path = os.path.join(scene_save_root, f"scene_{scene_idx}.log")
        with open(scene_save_path, "a") as scene_f:
            text = b64decode(
                eval(
                    requests.post(
                        env_url + "/reset", json={"json_file": None}
                    ).text
                )
            )

            obs, infos = loads(text)
            admissible_commands = format_admissible_commands(
                infos["admissible_commands"][0]
            )
            initial_obs, task_desc = format_obs_task_desc(obs)
            if initial_obs == "" or task_desc == "":
                continue
            scene_f.write(
                f"---------------------------------------------------------Scene: {scene_idx}---------------------------------------------------------\n"
            )
            scene_f.write(
                f"--------------------------------------------------------------------------------------------------------------------------------------------------------------------\n"
            )
            scene_f.write(
                f"--------------------------------------------------------------------------------------------------------------------------------------------------------------------\n"
            )

            print(f"Task Description: {task_desc}")
            images = deque(maxlen=args.max_images)
            for step in range(max_step):
                image_root = os.path.join(scene_save_root, f"images")
                if not os.path.exists(image_root):
                    os.makedirs(image_root)
                image_path = os.path.join(image_root, f"scene_{scene_idx}_{step}.jpg")
                image = get_image(env_url, args.is_ins_seg)
                images.append(image)
                image.save(image_path, "JPEG")

                scene_f.write(
                    f"Step:--------------------------------------------------------------------------{step+1}--------------------------------------------------------------------------\n"
                )

                # Use VLM to get the current admissble action
                prompt = generate_action_prompt(
                    task_desc, admissible_commands, task_hint, len(images)
                )
                scene_f.write(f"VLM Prompt:\n{prompt}\n")

                messages = {
                    "text": prompt,
                    "images": list(images),
                }

                start_time = time.time()
                for tried in range(5):
                    action_vlm_decoding_args = DecodingArguments(
                        max_tokens=8192,
                        n=1,
                        temperature=0.5 + (tried) * 0.1,
                        image_detail="auto",
                    )
                    action = actor_vlm_model.call_model(
                        messages,
                        decoding_args=action_vlm_decoding_args,
                        return_list=False,
                    ).strip()
                    scene_f.write(
                        f"\nOriginal VLM Response:\n{action}\n\n"
                    )
                    action = refine_action(action).strip().replace("\n", "")
                    if "No action" != action:
                        break
                end_time = time.time()
                scene_f.write(
                    f"> Action: {action}\nRunning time: {end_time - start_time} seconds\n\n"
                )
                print(f">> Action is: {action}")

                # Interact with envrionment and get resulting observation for the chosen action
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

                admissible_commands = format_admissible_commands(admissible_commands)
                succeed = done
                goal_condition_success_rate = infos["goal_condition_success_rate"][0]

                # Log SUCCEED task and the Goal condition success rate if the task is SUCCEED
                if succeed:
                    image_path = os.path.join(
                        image_root, f"scene_{scene_idx}_{step+1}.jpg"
                    )
                    image = get_image(env_url, args.is_ins_seg)
                    image.save(image_path, "JPEG")
                    with open(succeed_num_save_path, "a") as succeed_num_f:
                        succeed_num_f.write(
                            f"{scene_idx} Path:{json_file}: SUCCEED, Goal condition success rate: {goal_condition_success_rate}\n"
                        )
                        succeed_num_f.close()
                    num_succeess += 1
                    print("SUCCEED")
                    scene_f.write("SUCCEED\n")
                    break

            # Log UNSUCCEED task and the Goal condition success rate if the task is UNSUCCEED
            if not succeed:
                with open(succeed_num_save_path, "a") as succeed_num_f:
                    print("UNSUCCEED")
                    succeed_num_f.write(
                        f"{scene_idx} Path:{json_file}: UNSUCCEED, Goal condition success rate: {goal_condition_success_rate}\n"
                    )
                    succeed_num_f.close()
                scene_f.write("UNSUCCEED\n")
        scene_f.close()

    total_money = get_total_money()
    print(f"Total Money Cost: {total_money}")

    # Compute the success rate of total sample scenes
    with open(succeed_num_save_path, "a") as succeed_num_f:
        succeed_num_f.write(f"\nTotal number: {len(json_file_list)}\n")
        succeed_num_f.write(f"Total succeed number: {num_succeess}\n")
        succeed_num_f.write(f"Total Money Cost: {total_money}\n")
        succeed_num_f.close()

    requests.post(env_url + "/close", json={})

