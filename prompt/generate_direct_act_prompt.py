import json
import os

CURRENT_FOLDER = os.path.dirname(os.path.abspath(__file__))

basic_action_prompt = """
**Objective:**
Given images depicting complex scenes with multiple elements, your objective is to analyze scenes of the image about the specified task description and choose the appropriate action from a list of available actions to accomplish the given task description.

**Task Description and Instructions:**
{task_desc} {task_hint}

**Available Actions:**
{admissible_commands}

**Response:**
- Let's think step-by-step and choose the appropriate action to accomplish the task description, as the form [Begin][Selected Action][End] (e.g. [Begin]2. go to kitech 1[End]).
"""

def to_json():
   saved_action_prompt = basic_action_prompt[1:]
   saved_action_prompt = saved_action_prompt[:-1]
   
   json_dic = {
      "pick_and_place_simple": "The agent must find an object of the desired type, pick it up, find the correct location to place it, and put it down there.",
      "look_at_obj_in_light": "The agent must find an object of the desired type, locate and turn on a light source with the desired object in-hand.",
      "pick_clean_then_place_in_recep": "The agent must find an objectof the desired type, pick it up, go to a sink or a basin, clean the object with a sink or basin, then find the correct location to place it, and put it down there.",
      "pick_heat_then_place_in_recep": "The agent must find an object of the desired type, pick it up, go to a microwave, heat the object with the microwave, then find the correct location to place it, and put it down there.",
      "pick_cool_then_place_in_recep": "The agent must find an object of the desired type, pick it up, go to a fridge, cool the object with the fridge, then find the correct location to place it, and put it down there.",
      "pick_two_obj_and_place": "The agent must find an object of the desired type, pick it up, find the correct location to place it, put it down there, then look for another object of the desired type, pick it up, return to previous location, and put it down there with the other object.",
      "basic_action_prompt": saved_action_prompt,
   }
   json_str = json.dumps(json_dic, indent=4)
   
   file_path = os.path.join(CURRENT_FOLDER, "direct_act_prompt.json")
   
   with open(file_path, 'w') as file:
      file.write(json_str)
   
   print(f'Dictionary has been saved to {file_path}')
   

if __name__ == "__main__":
   print("Running Save to JSON.")
   to_json()
