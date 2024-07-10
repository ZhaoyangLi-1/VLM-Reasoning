from agi.utils import set_gpus
from agi.utils.chatbot_utils import ChatBot, DecodingArguments
from PIL import Image
import argparse

# ACTION_PROMPT = """**Task**
# **Task**
# Please perform as an embodied AI agent. Our final objective is to {task_desc}. There are two admissible actions for the next step and the current observation. Please determine the best action for the next step!

# **Current Observation**
# Displayed is an image capturing the current position of the agent.
# <image>

# **Admissible Actions List**
# {admissible_commands}

# **Analysis Guideline**
# 1. Determine if "{task_related_object}" is in the current observation.
# 2. If the current observation contains "{task_related_object}", pick it up. If it doesn't, we need to search for other locations.
# 3. Choose the most appropriate action from the list of admissible actions.
# 3. Summarize the analysis using the JSON format:
# {{
#     "thought":"the summarization of the analysis", 
#     "action": "the most appropriate action"
# }}

# Please select the most suitable action by following the analysis guideline."""


ACTION_PROMPT = """<image>
Please act as an embodied AI agent. Our objective is to {task_desc}.You are observing the current status of the task, and there are two admissible actions: {admissible_commands} for the next step. You must choose one action from the two admissible actions for the next step!

Your response should be json file in the following format: 
{{Thought: first determine if "{task_related_object}" is in the current observation, then think carefully about the two admissible actions. 
Action: choose one appropriate action
}}"""

def main(args):
    set_gpus(num_gpus=2)
        
    decoding_arguments = DecodingArguments(temperature=0.6, max_tokens=8192)
    chatbot = ChatBot(args.model, use_cpp=True)

    image = Image.open("/root/put-image.png")
    image = image.resize((300, 300))
    prompt = ACTION_PROMPT.format(
        task_desc="put some alarmclock on desk",
        admissible_commands="[go to garbagecan 1, put alarmclock 1 in/on desk 1]",
        task_related_object="alarmclock"
    )
    print(f"Prompt:\n{prompt}")
    inputs = {"images": [image], "text": prompt}

    response = chatbot.call_model(inputs, decoding_arguments)

    print(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llava-1.6-vicuna-13b")
    args = parser.parse_args()
    main(args)