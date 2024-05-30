import sys
import os

# Retrieve the path from the environment variable
llava_root = os.getenv('LLAVA_ROOT')

if not llava_root:
    raise EnvironmentError("LLAVA_ROOT environment variable is not set.")
if not os.path.isdir(llava_root):
    raise EnvironmentError(f"LLAVA_ROOT directory {llava_root} does not exist.")

# Print LLAVA_ROOT
print(f"LLAVA_ROOT is set to: {llava_root}")

# Add the scripts directory to the Python path
scripts_path = os.path.join(llava_root, 'scripts')
sys.path.append(scripts_path)

# Debug: Print system path
from chatbot_utils import ChatBotLLaVA, DecodingArguments

# Define decoding arguments
action_vlm_decoding_args = DecodingArguments(
    max_tokens=8192, n=1, temperature=0.7, image_detail="auto"
)

# Initialize the ChatBot model
actor_vlm_model = ChatBotLLaVA("llava-v1.5-7b", action_vlm_decoding_args)

print("ChatBot model initialized successfully.")
