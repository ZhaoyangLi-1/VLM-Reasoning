import os
from scripts.chatbot_utils import ChatBot, DecodingArguments



action_vlm_decoding_args = DecodingArguments(
    max_tokens=8192,
    n=1,
    temperature=0.7,
    image_detail="auto",
    )
actor_vlm_model = ChatBot("llava-v1.6-13b", action_vlm_decoding_args)