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

CURRENT_FOLDER = os.path.dirname(os.path.abspath(__file__))
ALFWORLD_DATA = os.getenv("ALFWORLD_DATA")


# Save images to a video
def save_video(frames: Sequence[np.ndarray], filename: str, fps: int = 3):
    frames =  [np.array(image) for image in frames]
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(filename, codec='libx264', audio=False)

def get_task_hints(task_hints_prompt_path):
    task_hints = {}
    for filepath, _, filenames in os.walk(task_hints_prompt_path):
        for filename in filenames:
            take_name = filename.split(".")[0]
            with open(os.path.join(filepath, filename), "r") as f:
                task_hints[take_name] = f.read().replace("\n", " ")
    return task_hints


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
def draw_instance_img(original_image, env_url):
    def is_dark_color(color):
        r, g, b, _ = color
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

    original_image = original_image.convert("RGBA")
    objects_receps = loads(
        b64decode(eval(requests.post(env_url + "/get_objects_receps", json={}).text))
    )[0]
    instance_segs_list, instance_detections2D_list = loads(
        b64decode(
            eval(requests.post(env_url + "/get_instance_seg_and_id", json={}).text)
        )
    )

    transparency = 0.2
    segment, instance_detections2D = (
        instance_segs_list[0],
        instance_detections2D_list[0],
    )
    if not isinstance(segment, Image.Image):
        segment = Image.fromarray(np.array(segment))
        segment = segment.convert("RGBA")
        segment.putalpha(int(255 * transparency))

    combined = Image.new("RGBA", original_image.size)
    combined.paste(original_image, (0, 0))
    combined.paste(segment, (0, 0), segment)
    draw = ImageDraw.Draw(combined)
    font = ImageFont.load_default() #ImageFont.truetype("/home/zhaoyang/.fonts/Helvetica.ttf", size=50) 

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

    # Second pass to draw text and overlay
    for obj_text_id, (bbox, num_inside) in obj_dic.items():
        x1, y1, x2, y2 = bbox

        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width, text_height = (
            text_bbox[2] - text_bbox[0],
            text_bbox[3] - text_bbox[1],
        )

        # Center the text in the bounding box
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        text_x = center_x - text_width / 2 + 40
        text_y = center_y - text_height / 2

        if num_inside != 0:
            text_x = x1 + 1 / num_inside * 50
            text_y = y1 + 1 / num_inside * 20

        new_pos = (text_x, text_y, text_x + text_width, text_y + text_height)
        while is_overlapping(new_pos, drawn_text_positions):
            text_x += 10
            text_y += 10
            new_pos = (text_x, text_y, text_x + text_width, text_y + text_height)

        avg_bg_color = get_average_color(combined, new_pos)
        text_color = "white" if is_dark_color(avg_bg_color) else "black"
        draw.text((text_x, text_y), obj_text_id, fill=text_color)
        drawn_text_positions.append(new_pos)

    combined = combined.convert("RGB")
    return combined


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
        return obs_desc, task_desc


# Delete "(i): [Selected Action]" to just "[Selected Action]"
def refine_action(response):
    # Split the response by "**Response:**" to separate the header from the actions
    parts = response.split('**Response:**')
    actions = parts[1] if len(parts) > 1 else response
    # Regular expression to find all matches of the pattern "optional leading characters (number): some action"
    # matches = re.findall(r"[-\s]*\(\d+\): ([^\n]+)", actions)
    matches = re.findall(r"\[BEGIN\]\((\d+)\): ([^\[]+)\[END\]", actions)
    if not matches:
        return "No action"
    # Extract and return the first action if any matches are found
    first_action = matches[0][1].strip() if matches else "No action"
    return first_action

# def refine_action(response):
#     # Regular expression to match the pattern "(i): some action"
#     match = re.search(r"\(\d+\): ([^\.]+)", response)
#     # Extract and return the action if the match is found
#     return match.group(1).strip() if match else "No action"


# Get current image form envrionment
def get_image(env_url, is_ins_seg):
    text = b64decode(eval(requests.post(env_url + "/get_frames", json={}).text))
    image = loads(text).squeeze()[:, :, ::-1].astype("uint8")
    image = Image.fromarray(image)
    if is_ins_seg:
        image = draw_instance_img(image, env_url)
    return image


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
