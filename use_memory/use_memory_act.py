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

def get_prompts():
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
    vlm_prompt_for_one_img_path = os.path.join(PROMPT_PATH, "use-memory/one_image.txt")
    with open(vlm_prompt_for_one_img_path, "r") as f:
        vlm_prompt_for_one_img = f.read()
    summary_prompt_path = os.path.join(PROMPT_PATH, "use-memory/summerize_the_analysis.txt")
    with open(summary_prompt_path, "r") as f:
        summary_promp = f.read()
    extract_related_objects_path = os.path.join(PROMPT_PATH, "use-memory/extract_related_objects.txt")
    with open(extract_related_objects_path, "r") as f:
        extract_related_objects_prompt = f.read()
    return generate_plan_prompt, vlm_prompt_for_one_img, summary_promp, task_hints, extract_related_objects_prompt


def delete_inefficient_action(admissible_commands, no_try_actions):
    return [element for element in admissible_commands if element not in no_try_actions]


def test_tasks(args):
    generate_plan_prompt, vlm_prompt_for_one_img, summary_promp, task_hints, extract_related_objects_prompt = get_prompts()
    env_url = "http://127.0.0.1:" + str(args.env_url)
    # initial VLM and LLM model
    if "gpt" in args.vlm_model:
        os.environ["AGI_ROOT"] = "/home/zhaoyang/projects/neural-reasoning"
        sys.path.append(os.path.join(os.environ["AGI_ROOT"]))
        # from agi.utils.openai_utils import get_total_money
        from agi.utils.chatbot_utils import DecodingArguments, ChatBot
        action_vlm_decoding_args = DecodingArguments(
            max_tokens=8192,
            n=1,
            temperature=0.7,
            image_detail="auto",
            )
        actor_vlm_model = ChatBot(args.vlm_model)
    elif "llava" in args.vlm_model:
        sys.path.append("/home/zhaoyang/projects/LLaVA//llava")
        from llava.chatbot_utils import ChatBot, DecodingArguments
        action_vlm_decoding_args = DecodingArguments(
            max_tokens=8192,
            n=1,
            temperature=0.7,
            image_detail="auto",
            use_4bit=True
        )
        actor_vlm_model = ChatBot(args.vlm_model, args.model_base)
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
        # first one is initial observation
        # ini_obs = refine_ini_obs(ini_obs)
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
            count_dic, instance_image, image, merged_obj_dic = get_obj_infor(env_url, 0)
            image.save(image_path, "JPEG")
            instance_image.save(instance_image_path, "JPEG")
            images_queue.append(image)
            images_log.append(image)
            image_paths_queue.append(image_path)
            
            # if all_history is None:
            #     all_history = ""
            #     one_image_prompt = vlm_prompt_for_one_img.format(task_description=task_desc, ini_obs=ini_obs, plan=plan, history="No history.", admissible_commands=admissible_commands)
            # else:
            #     one_image_prompt = vlm_prompt_for_one_img.format(task_description=task_desc, ini_obs=ini_obs, plan=plan, history=all_history, admissible_commands=admissible_commands)
                
            if all_history is None:
                all_history = f"State {0}:\nNo history.\n"
                one_image_prompt = vlm_prompt_for_one_img.format(task_description=task_desc, plan=plan, history="No history.", admissible_commands=admissible_commands)
            else:
                one_image_prompt = vlm_prompt_for_one_img.format(task_description=task_desc, plan=plan, history=all_history, admissible_commands=admissible_commands)
            
            # object_names = list(merged_obj_dic.keys())
            # all_objects = object_names + task_related_objects
            # true_task_related_objects_exist_dic = {obj: obj in object_names for obj in task_related_objects}
            # predicted_result_for_task_related_dic = {key: None for key in true_task_related_objects_exist_dic.keys()}
            # for obj_name in all_objects:
            #     print(f"Current Object for Detection: {obj_name}")
            #     true_count = 1
            #     if obj_name in true_task_related_objects_exist_dic and true_task_related_objects_exist_dic[obj_name] is False:
            #         true_count = 0
            #     if all_history is None:
            #         obj_existence_prompt = existence_prompt.format(history="No history.", obj_name=obj_name)
            #     else:
            #         obj_existence_prompt = existence_prompt.format(history=all_history, obj_name=obj_name)
            #     messages = {"text": obj_existence_prompt, "images": list(images_queue)}
            #     existence_res = actor_vlm_model.call_model(messages, decoding_args=action_vlm_decoding_args, return_list=False).strip()
            #     print(f"VLM Existence Response:{existence_res}")
            #     if obj_name in task_related_objects:                    
            #         if (true_task_related_objects_exist_dic[obj_name] is True and "Yes" in response) or (true_task_related_objects_exist_dic[obj_name] is False and "No" in existence_res):
            #             counts_hat = 1
            #             correctness = "correct"
            #         elif (true_task_related_objects_exist_dic[obj_name] is True and "No" in response) or (true_task_related_objects_exist_dic[obj_name] is False and "Yes" in existence_res):
            #             counts_hat = 0
            #             correctness = "incorrect"
            #         obj_name = f"desired_{obj_name}"
            #     else:
            #         if "Yes" in existence_res:
            #             counts_hat = 1
            #             correctness = "correct"
            #         else:
            #             counts_hat = 0
            #             correctness = "incorrect"
            #     print(f"object_category: {obj_name}, true_counts: {true_count}, predicted_counts: {counts_hat}, correctness: {correctness}, history_num: {step}")
            #     answer.append([obj_name, true_count, counts_hat, correctness, image_path, step])
            
            
            messages = {"text": one_image_prompt,"images": list(images_queue)}
            start_time = time.time()
            response = actor_vlm_model.call_model(messages, decoding_args=action_vlm_decoding_args, return_list=False).strip()
            end_time = time.time()
            print(f"Response:\n{response}")
            action = refine_action(response)
            print(f"Chosen Action: {action}")
            if "No action" in action:
                # breakpoint()
                continue
            no_try_actions.append(action)
            summary_prompt_for_one_img = summary_promp.format(context=response)
            messages = {"text": summary_prompt_for_one_img}
            
            data.append([task_idx, task_desc, step+1, action, image_path, instance_image_path, task_related_objects, count_dic, merged_obj_dic, current_num_history, all_history])
            
            history = llm_model.call_model(messages, decoding_args=action_vlm_decoding_args, return_list=False).strip()
            current_num_history = len(history_steps)
            history_steps.append(f"State {current_num_history}:\n{history}\n")
            all_history = ''.join(history_steps)
            
            # data.append(["task_desc", "action", "original_image_path", "ins_image_path", "object_count_infor", "object_bboxes", "history_num", "history"])
            
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
                # get_total_money(),
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
                image = get_image(env_url, False, args.is_ins_seg)
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
            
    csv_save_folder = os.path.join(CURRENT_FOLDER, args.result_csv_folder, args.refine_type)
    if not os.path.exists(csv_save_folder):
        os.makedirs(csv_save_folder)
    csv_save_path = os.path.join(csv_save_folder, f"llava-aflworld-history-infor.csv")
    write_data_to_csv(csv_save_path, data)
    # total_money = get_total_money()
    total_tasks_log.write(f"\nTotal number: {len(json_file_list)}\n")
    total_tasks_log.write(f"Total succeed number: {num_succeess}\n")
    # total_tasks_log.write(f"Total money Cost: {total_money}\n")

    requests.post(env_url + "/close", json={})
            
            # a=1
# '**Analysis**\n\n1. As there is no history available, we are at the first step in the plan and our current place is "Unknown".\n2. In the current observation, we can see a sidetable with a desklamp and an alarm clock on it.\n3. Based on the requirements for the current step, which is to find an alarm clock, we can confirm that we have already found it as it is visible on the sidetable in the current observation.\n4. The most appropriate action to take next is to pick up the alarm clock, but the admissible actions do not include any "pick up" actions. Therefore, the next best action to take is to "go to sidetable 1" to be closer to the alarm clock and prepare for the next step in the plan.\n5. The most suitable action to take next is: (2) go to sidetable 1.'