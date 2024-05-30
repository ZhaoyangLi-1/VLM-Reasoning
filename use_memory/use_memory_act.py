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


ALFWORLD_DATA = os.getenv("ALFWORLD_DATA")
ALFWORLD_SAVE = os.getenv("ALFWORLD_SAVE")
CURRENT_FOLDER = os.path.dirname(os.path.abspath(__file__))
PARENT_FOLDER = os.path.dirname(CURRENT_FOLDER)
PROMPT_PATH = os.path.join(PARENT_FOLDER, "prompts/alfworld")


def write_data_to_csv(filename, data):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        for row in data:
            writer.writerow(row)
            print(f'Added row: {row}')

def get_prompts(is_generate_object_list=False):
    def get_task_hints(task_hints_prompt_path):
        task_hints = {}
        for filepath, _, filenames in os.walk(task_hints_prompt_path):
            for filename in filenames:
                take_name = filename.split(".")[0]
                with open(os.path.join(filepath, filename), "r") as f:
                    task_hints[take_name] = f.read().replace("\n", " ")
        return task_hints
    task_hints_prompt_path = os.path.join(PROMPT_PATH, "task-hints")
    task_hints = get_task_hints(task_hints_prompt_path)
    
    generate_plan_prompt_path = os.path.join(PROMPT_PATH, "use-memory/generate_plan.txt")
    with open(generate_plan_prompt_path, "r") as f:
        generate_plan_prompt = f.read()
    
    if is_generate_object_list:
        vlm_prompt_for_one_img_path = os.path.join(PROMPT_PATH, "use-memory/one_image_with_object_list.txt")
    else:
        vlm_prompt_for_one_img_path = os.path.join(PROMPT_PATH, "use-memory/one_image.txt")
    with open(vlm_prompt_for_one_img_path, "r") as f:
        vlm_prompt_for_one_img = f.read()
        
    summary_prompt_path = os.path.join(PROMPT_PATH, "use-memory/summerize_the_analysis.txt")
    with open(summary_prompt_path, "r") as f:
        summary_promp = f.read()
        
    extract_related_objects_path = os.path.join(PROMPT_PATH, "use-memory/extract_related_objects.txt")
    with open(extract_related_objects_path, "r") as f:
        extract_related_objects_prompt = f.read()
        
    generate_object_list_path = os.path.join(PROMPT_PATH, "use-memory/list_objects.txt")
    with open(generate_object_list_path, "r") as f:
        generate_object_list_prompt = f.read()
    
    return generate_plan_prompt, vlm_prompt_for_one_img, summary_promp, task_hints, extract_related_objects_prompt, generate_object_list_prompt


def delete_inefficient_action(admissible_commands, no_try_actions):
    return [element for element in admissible_commands if element not in no_try_actions]


def test_tasks(args):
    generate_plan_prompt, vlm_prompt_for_one_img, summary_promp, task_hints, extract_related_objects_prompt, generate_object_list_prompt = get_prompts(args.is_generate_object_list)
    env_url = "http://127.0.0.1:" + str(args.env_url)
    # initial VLM and LLM model
    action_vlm_decoding_args = DecodingArguments(
        max_tokens=8192,
        n=1,
        temperature=0.7,
        image_detail="auto",
        )
    actor_vlm_model = ChatBot(args.vlm_model)

    print(f"VLM Model: {args.vlm_model}")
    # Setup VLM(gpt4-v) model as action selector and current attempts relfector
    llm_decoding_args = DecodingArguments(
        max_tokens=1024,
        n=1,
        temperature=0.7,
        image_detail="auto",
    )
    llm_model = ChatBot(args.llm_model)
    num_succeess = 0
    save_path = os.path.join(ALFWORLD_SAVE, args.save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    total_tasks_log = Global_Logger(
        os.path.join(save_path, f"succeed-num-begin-{args.begin_task}.log")
    )

    json_file_list = total_tasks_log.skip_succeed(PARENT_FOLDER, args)
    set_dic = {"env_type": "visual", "batch_size": 1}
    requests.post(env_url + "/set_environment", json=set_dic).text
    data = []
    data.append(["task_id", "task_desc", "step", "action", "original_image_path", "ins_image_path", "task_related_objects", "object_count_infor", "object_bboxes", "history_num", "history"])
    for task_idx, json_file in enumerate(json_file_list):
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
        receps, task_desc, ini_obs = format_initial_obs_to_get_rceps(obs)
        admissible_commands = delete_examine_action_for_receps(infos["admissible_commands"][0], receps)
        admissible_commands = format_admissible_commands(admissible_commands)
        # generate plan
        plan_prompt = generate_plan_prompt.format(task_explanation=task_hint, task=task_desc)
        messages = {"text": plan_prompt}
        plan = llm_model.call_model(messages, decoding_args=llm_decoding_args, return_list=False).strip()
        
        extract_related_objects_messages = extract_related_objects_prompt.format(task_desc=task_desc)
        messages = {"text": extract_related_objects_messages}
        task_related_objects = ""
        while len(task_related_objects) ==0:
            task_related_objects = llm_model.call_model(messages, decoding_args=llm_decoding_args, return_list=False).strip()
            print(f"Task Related Objects: {task_related_objects}")
        task_related_objects = ast.literal_eval(task_related_objects)
        # Set image_queue
        images_queue = deque(maxlen=args.max_images)
        image_paths_queue = deque(maxlen=args.max_images)
        no_try_actions = deque(maxlen=3)
        all_history = None
        images_log = []
        
        history_steps = []
        history_steps.append(f"State {0}:\nNo history.\n")
        current_num_history = 0
        for step in range(args.max_step):
            # print(f"Current step: {step}")
            image_root = os.path.join(task_save_root, f"images")
            if not os.path.exists(image_root):
                os.makedirs(image_root)
            image_path = os.path.join(image_root, f"task-{task_idx}-{step}.jpg")
            instance_image_path = os.path.join(image_root, f"task-{task_idx}-{step}-ins-seg.jpg")
            # image = get_image(env_url, args.is_ins_seg)
            count_dic, instance_image, image, merged_obj_dic = get_obj_infor(env_url)
            image.save(image_path, "JPEG")
            instance_image.save(instance_image_path, "JPEG")
            images_queue.append(image)
            images_log.append(image)
            image_paths_queue.append(image_path)
            
            if args.is_generate_object_list:
                messages = {"text": generate_object_list_prompt,"images": list(images_queue)}
                object_list = actor_vlm_model.call_model(messages, decoding_args=action_vlm_decoding_args, return_list=False).strip()
                if all_history is None:
                    all_history = f"State {0}:\nNo history.\n"
                    one_image_prompt = vlm_prompt_for_one_img.format(object_list=object_list, task_description=task_desc, plan=plan, history="No history.", admissible_commands=admissible_commands)
                else:
                    one_image_prompt = vlm_prompt_for_one_img.format(object_list=object_list, task_description=task_desc, plan=plan, history=all_history, admissible_commands=admissible_commands)
            else:
                if all_history is None:
                    all_history = f"State {0}:\nNo history.\n"
                    one_image_prompt = vlm_prompt_for_one_img.format(task_description=task_desc, plan=plan, history="No history.", admissible_commands=admissible_commands)
                else:
                    one_image_prompt = vlm_prompt_for_one_img.format(task_description=task_desc, plan=plan, history=all_history, admissible_commands=admissible_commands)
            
            
            messages = {"text": one_image_prompt,"images": list(images_queue)}
            start_time = time.time()
            response = actor_vlm_model.call_model(messages, decoding_args=action_vlm_decoding_args, return_list=False).strip()
            end_time = time.time()
            action = refine_action(response)
            if "No action" in action:
                continue
            no_try_actions.append(action)
            summary_prompt_for_one_img = summary_promp.format(context=response)
            messages = {"text": summary_prompt_for_one_img}
            
            # data.append(["task_id", "task_desc", "step", "action", "original_image_path", "ins_image_path", "task_related_objects", "object_count_infor", "object_bboxes", "history_num", "history"])
            print(f"task_idx: {task_idx}, task_desc: {task_desc}, step: {step+1}, action: {action}, count_dic: {count_dic},  current_num_history: {current_num_history}")
            data.append([task_idx, task_desc, step+1, action, image_path, instance_image_path, task_related_objects, count_dic, merged_obj_dic, current_num_history, all_history])
            history = llm_model.call_model(messages, decoding_args=action_vlm_decoding_args, return_list=False).strip()
            current_num_history = len(history_steps)
            history_steps.append(f"State {current_num_history}:\n{history}\n")
            all_history = ''.join(history_steps)
            
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
                one_image_prompt,
                response,
                action,
                start_time,
                end_time,
                0,
                summary_prompt_for_one_img,
                history,
                obs,
            )
            admissible_commands = delete_examine_action_for_receps(admissible_commands, receps)
            admissible_commands = delete_inefficient_action(admissible_commands, no_try_actions)
            admissible_commands = format_admissible_commands(
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
            
    csv_save_folder = os.path.join(PARENT_FOLDER, "test-detect")
    if not os.path.exists(csv_save_folder):
        os.makedirs(csv_save_folder)
    csv_save_path = os.path.join(csv_save_folder, f"{args.vlm_model}-add-object-list-aflworld-history-infor.csv")
    print(f"Save path: {csv_save_path}")
    write_data_to_csv(csv_save_path, data)
    total_tasks_log.write(f"Total succeed number: {num_succeess}\n")

    requests.post(env_url + "/close", json={})