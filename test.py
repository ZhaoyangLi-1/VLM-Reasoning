# import os
# import sys
# sys.path.append("/home/zhaoyang/projects/LLaVA//llava")
# from llava.chatbot_utils import ChatBot, DecodingArguments

# def main():
#     action_vlm_decoding_args = DecodingArguments(
#             max_tokens=8192,
#             n=1,
#             temperature=0.7,
#             image_detail="auto",
#             use_4bit=True
#         )
#     actor_vlm_model = ChatBot("llava-v1.6-34b", None, action_vlm_decoding_args)
#     a = 1


# if __name__ == "__main__":
#     main()


import os
import sys
os.environ["AGI_ROOT"] = "/home/zhaoyang/projects/neural-reasoning"
sys.path.append(os.path.join(os.environ["AGI_ROOT"]))
from agi.utils.chatbot_utils import DecodingArguments, ChatBot
action_vlm_decoding_args = DecodingArguments(
    max_tokens=8192,
    n=1,
    temperature=0.7,
    image_detail="auto",
    )
actor_vlm_model = ChatBot("llava-v1.5-13b", action_vlm_decoding_args)
a=1