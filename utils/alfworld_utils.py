import json
import os
from base64 import b64decode
from pickle import loads
import requests
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import re
import pickle
from moviepy.editor import ImageSequenceClip
from typing import Sequence
from collections import deque, defaultdict
import random

CURRENT_FOLDER = os.path.dirname(os.path.abspath(__file__))
ALFWORLD_DATA = os.getenv("ALFWORLD_DATA")

# RAW_ADMISSIBLE_COMMANDS ={
#     "go to": "go to {recep}",
#     "take": "take {obj} from {recep}",
#     "put": "put {obj} in {recep}",
#     "clean": "clean {obj} with {recep}",
#     "heat": "heat {obj} with {recep}",
#     "cool": "cool {obj} with {recep}",
#     "examine": "examine {obj}",
#     "open": "open {obj}",
#     "close": "close {obj}",
# }


# Save images to a video
def save_video(frames: Sequence[np.ndarray], filename: str, fps: int = 3):
    frames =  [np.array(image) for image in frames]
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(filename, codec='libx264', audio=False, fps=fps)


# Get the tasks and return the task list
def get_path_tasks(task_list_path):
    task_paths = []
    with open(task_list_path, "rb") as f:
        relative_path_json = json.load(f)
        for _, rel_paths in relative_path_json.items():
            for rel_path in rel_paths:
                full_path = os.path.join(ALFWORLD_DATA, rel_path)
                task_paths.append(full_path)
    return sorted(task_paths)    


# Generate sgement image with instance object name
# def draw_instance_img(original_image, env_url):
#     def is_dark_color(color):
#         r, g, b, _ = color
#         return (0.299 * r + 0.587 * g + 0.114 * b) < 128

#     def is_inside(inner_bbox, outer_bbox):
#         return (
#             outer_bbox[0] <= inner_bbox[0] <= outer_bbox[2]
#             and outer_bbox[0] <= inner_bbox[2] <= outer_bbox[2]
#             and outer_bbox[1] <= inner_bbox[1] <= outer_bbox[3]
#             and outer_bbox[1] <= inner_bbox[3] <= outer_bbox[3]
#         )

#     def count_inner_boxes(obj_dic):
#         result = {}
#         for obj_id, bbox in obj_dic.items():
#             num_inside = 0
#             for other_id, other_bbox in obj_dic.items():
#                 if obj_id != other_id and is_inside(other_bbox, bbox):
#                     num_inside += 1
#             result[obj_id] = (bbox, num_inside)
#         return result

#     def get_average_color(image, bbox):
#         x1, y1, x2, y2 = bbox
#         area = image.crop((x1, y1, x2, y2))
#         avg_color = np.array(area).mean(axis=(0, 1))
#         return avg_color

#     def is_overlapping(new_pos, existing_positions):
#         x1_new, y1_new, x2_new, y2_new = new_pos
#         for pos in existing_positions:
#             x1, y1, x2, y2 = pos
#             if not (x2_new < x1 or x2 < x1_new or y2_new < y1 or y2 < y1_new):
#                 return True
#         return False

#     original_image = original_image.convert("RGBA")
#     objects_receps = loads(
#         b64decode(eval(requests.post(env_url + "/get_objects_receps", json={}).text))
#     )[0]
#     instance_segs_list, instance_detections2D_list = loads(
#         b64decode(
#             eval(requests.post(env_url + "/get_instance_seg_and_id", json={}).text)
#         )
#     )

#     transparency = 0.2
#     segment, instance_detections2D = (
#         instance_segs_list[0],
#         instance_detections2D_list[0],
#     )
#     if not isinstance(segment, Image.Image):
#         segment = Image.fromarray(np.array(segment))
#         segment = segment.convert("RGBA")
#         segment.putalpha(int(255 * transparency))

#     combined = Image.new("RGBA", original_image.size)
#     combined.paste(original_image, (0, 0))
#     combined.paste(segment, (0, 0), segment)
#     draw = ImageDraw.Draw(combined)
#     font = ImageFont.load_default() #ImageFont.truetype("/home/zhaoyang/.fonts/Helvetica.ttf", size=50) 

#     drawn_text_positions = []
#     obj_dic = {}
#     for obj_id, obj in objects_receps.items():
#         if obj_id in instance_detections2D:
#             bbox = instance_detections2D[obj_id].tolist()
#         else:
#             continue
#         text = str(obj["num_id"])
#         obj_dic[text] = bbox

#     obj_dic = count_inner_boxes(obj_dic)

#     # Second pass to draw text and overlay
#     for obj_text_id, (bbox, num_inside) in obj_dic.items():
#         x1, y1, x2, y2 = bbox

#         text_bbox = draw.textbbox((0, 0), text, font=font)
#         text_width, text_height = (
#             text_bbox[2] - text_bbox[0],
#             text_bbox[3] - text_bbox[1],
#         )

#         # Center the text in the bounding box
#         center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
#         text_x = center_x - text_width / 2 + 40
#         text_y = center_y - text_height / 2

#         if num_inside != 0:
#             text_x = x1 + 1 / num_inside * 50
#             text_y = y1 + 1 / num_inside * 20

#         new_pos = (text_x, text_y, text_x + text_width, text_y + text_height)
#         while is_overlapping(new_pos, drawn_text_positions):
#             text_x += 10
#             text_y += 10
#             new_pos = (text_x, text_y, text_x + text_width, text_y + text_height)

#         avg_bg_color = get_average_color(combined, new_pos)
#         text_color = "white" if is_dark_color(avg_bg_color) else "black"
#         draw.text((text_x, text_y), obj_text_id, fill=text_color)
#         drawn_text_positions.append(new_pos)

#     combined = combined.convert("RGB")
#     return combined

def draw_instance_img(original_image, env_url, image_size_margin_ratio=0.005):
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
        object_type = obj["object_type"]
        id = obj["num_id"].split(" ")[1]
        text = object_type + " " + id
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


# Get all json files
def get_json_files(data_path):
    train_data_list = []
    for filepath, _, filenames in os.walk(data_path):
        for filename in filenames:
            if filename.endswith("traj_data.json"):
                json_path = os.path.join(filepath, filename)
                train_data_list.append(json_path)
            if len(train_data_list) == 200:
                return train_data_list
    return train_data_list


def format_admissible_commands_refining_choices(admissible_commands, is_delete_go_to=False):
    def classify_objects(strings):
        object_dict = {}
        for s in strings:
            if "go to" in s:
                parts = s.split()
                object_name = parts[2]  # Assuming the object name is always the third word
                object_number = int(parts[3])  # Assuming the number is always the fourth word

                if object_name in object_dict:
                    object_dict[object_name].append(object_number)
                else:
                    object_dict[object_name] = deque([object_number])

        return object_dict
    
    def delete_go_to(strings):
        return [s for s in strings if "go to" not in s]
    
    def reduce_strings(strings):
        reduced_strings = set()
        for s in strings:
            if "go to" in s:
                parts = s.split()
                object_name = ' '.join(parts[:-1])  # Get all but the last element which is the number
                reduced_strings.add(object_name)
            else:
                reduced_strings.add(s)

        return list(reduced_strings)
    
    go_to_object_dic = None
    
    # Filter out "inventory" and "look"
    admissible_commands = [
        item for item in admissible_commands if item not in ["inventory", "look"]
    ]
    
    # Use regular expression to replace "sofai" with "sofa i"
    admissible_commands = [
        re.sub(r'sof(\d)', r'sofa \1', command) for command in admissible_commands
    ]
    
    if is_delete_go_to:
        admissible_commands = delete_go_to(admissible_commands)
    else:
        go_to_object_dic = classify_objects(admissible_commands)
    
    admissible_commands_formatted = "\n".join(
        f"({i + 1}): {s}" for i, s in enumerate(reduce_strings(admissible_commands))
    )
    
    return admissible_commands_formatted, go_to_object_dic


# Delte useless actions ["inventory", "look"] and formulate actions as (i): "[Selected Action]", where 'i' is the numerical position of the chosen action in the list.
def format_admissible_commands(admissible_commands):
    # Filter out "inventory" and "look"
    admissible_commands = [
        item for item in admissible_commands if item not in ["inventory", "look"]
    ]
    
    # Use regular expression to replace "sofai" with "sofa i"
    admissible_commands = [
        re.sub(r'sof(\d)', r'sofa \1', command) for command in admissible_commands
    ]
    
    # admissible_commands = reduce_strings(admissible_commands)
    # Format the commands
    admissible_commands_formatted = "\n".join(
        f"({i + 1}): {s}" for i, s in enumerate(admissible_commands)
    )
    return admissible_commands_formatted



# Get the done path
def get_done_paths(lines):
    paths = []
    for text in lines:
        import re

        match = re.search(r"Path:(.+?\.json)", text)
        if match:
            paths.append(match.group(1))
    return paths


# Format observation and tsk description
def format_obs_task_desc(obs):
    obs = obs[0].replace("\n\n", "\n")
    obs = obs.split("\n")
    if len(obs) < 2:
        return "", ""
    else:
        obs_desc = obs[-2]
        task_start = obs[-1].find(":") + 1
        task_desc = obs[-1][task_start:].strip()
        # task_desc = task_desc[0].upper() + task_desc[1:] + "."
        return obs_desc, task_desc



# def refine_action(response):
#     # Split the response to separate the header from the actions
#     parts = response.split('**Response:**')
#     actions = parts[1] if len(parts) > 1 else response
#     # Regular expression to find all matches of the pattern "[Begin](number): action [End]"
#     matches = re.findall(r"\[Begin\]\((\d+)\): ([^\[]+)\[End\]", actions, re.IGNORECASE)
#     if not matches:
#         return "No action"
#     # Combine the number with the action, separated by a period and a space
#     first_action = matches[0][1].strip() if matches else "No action"
#     return first_action

# def refine_action(response):
#     # Regular expression to match the pattern "(i): some action"
#     match = re.search(r"\(\d+\): ([^\.]+)", response)
#     # Extract and return the action if the match is found
#     return match.group(1).strip() if match else "No action"

def refine_action(response):
    # response = response.split("The answer is:")
    response = response.split("The Most Appropriate Action: ")
    if len(response) == 1:
        return "No action"
    response = response[1].strip()
    match = re.search(r"\(\d+\): ([^\.]+)", response)
    if match is None:
        action = response
    else:
        return match.group(1).strip()
    if not bool(re.search(r'\d$', action)):
        action = "No action"
    return action
    
def delete_examine_action_for_receps(admissible_commands, recep_names):
    return [command for command in admissible_commands if "examine" not in command or not any(name in command for name in recep_names)]


def format_initial_obs_to_get_rceps(obs):
    obs = obs[0].replace("\n\n", "\n").split("\n")
    obs_desc = obs[1]
    task_start_start_pos = obs[2].find(":") + 1
    task_desc = obs[2][task_start_start_pos:].strip()
    big_objects = obs_desc.split("you see a ")[1].split(", ")
    big_objects[-1] = big_objects[-1].replace("and a ", "")
    big_objects[-1] = big_objects[-1].replace(".", "")
    big_objects = [s.replace("a ", "") for s in big_objects]
    return big_objects, task_desc, obs_desc


# Get current image form envrionment
# def get_image(env_url, is_ins_seg):
#     text = b64decode(eval(requests.post(env_url + "/get_frames", json={}).text))
#     image = loads(text).squeeze()[:, :, ::-1].astype("uint8")
#     image = Image.fromarray(image)
#     if is_ins_seg:
#         image = draw_instance_img(image, env_url)
#     return image

def get_image(env_url, is_seg):
    text = b64decode(eval(requests.post(env_url + "/get_frames", json={}).text))
    image = loads(text).squeeze()[:, :, ::-1].astype("uint8")
    image = Image.fromarray(image)
    if is_seg:
        image, merged_obj_dic = draw_instance_img(image, env_url)
        return image, merged_obj_dic
    return image


def to_pascal_case(text):
    if not text:
        return text
    return text[0].upper() + ''.join(text[i].upper() if text[i-1].islower() else text[i] for i in range(1, len(text)))

def get_obj_infor(env_url):
    instance_image, merged_obj_dic = get_image(env_url, True)
    original_image = get_image(env_url, False)
    count_dict = {}
    for obj_type, bboxs in merged_obj_dic.items():
        count_dict[obj_type] = len(bboxs)

    return count_dict, instance_image, original_image, merged_obj_dic


def get_roate_ins_images(env_url, is_ins_seg, image_root):
    rotate_images_path = os.path.join(image_root, "initial_rotrate_images")
    if not os.path.exists(rotate_images_path):
        os.makedirs(rotate_images_path)
    rotate_images = []
    rotate_degrees = [0, 90, 180, 270]
    horizon = 0
    rotate_commands = [f"RotateRight_{degree}_{horizon}" for degree in rotate_degrees]
    for rotate_command in rotate_commands:
        requests.post(env_url + "/step_rotate", json={"action": rotate_command}).text
        image = get_image(env_url, is_ins_seg)
        rotate_images.append(image)
        image.save(os.path.join(rotate_images_path, f"{rotate_command}.png"))
    requests.post(env_url + "/step_to_original_rotation", json={}).text
    return rotate_images

def refine_ini_obs(ini_obs):
    head, obs = ini_obs.split('. ')
    items_with_numbers = re.findall(r'(\w+)\s(\d+)', obs)

    item_ranges = defaultdict(list)
    for item, number in items_with_numbers:
        item_ranges[item].append(int(number))
        
    formatted_items = []
    for item, numbers in item_ranges.items():
        numbers.sort()
        if len(numbers) == 1:
            formatted_items.append(f"{item} ({numbers[0]})")
        else:
            formatted_items.append(f"{item} ({numbers[0]}-{numbers[-1]})")

    formatted_text = ', '.join(formatted_items)
    return head + ". Looking quickly around you, you can see " + formatted_text
    

def generate_html_with_task_log(context, image_paths, output_file):
    html_dir = os.path.dirname(output_file)
    image_paths = [os.path.relpath(path, html_dir) for path in image_paths]


    content = context

    images_html = ''.join([f'<img src="{path}" alt="Image" style="max-width: 100%; height: auto;">' for path in image_paths])
    images_container = f'<div class="images-container">{images_html}</div>'
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Task Log</title>
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
        }}
        .images-container {{
            margin: 20px auto;
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
        }}
        .images-container img {{
            margin: 10px;
            max-width: 15%; 
            height: auto;
        }}
        .content-container {{
            text-align: left;
            margin: 20px;
            white-space: pre-wrap;
        }}
    </style>
</head>
<body>
    {images_container}
    <div class="content-container">{content}</div>
</body>
</html>
"""

    with open(output_file, 'w') as file:
        file.write(html)


        
a = "Analysis: The history information indicates that we are at step 1 of the plan, which is to find an alarm clock. The current observation shows us in a room with a sidetable and a desklamp. The sidetable has a drawer that could potentially contain an alarm clock. The objects relevant to our task in the current observation are the sidetable and the desklamp. Since we need to find an alarm clock, examining the sidetable where alarm clocks are commonly placed would be a logical next step.\n\nThe Most Appropriate Action: examine sidetable 1"
b = refine_action(a)