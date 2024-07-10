from agi.utils.chatbot_utils import DecodingArguments, ChatBot
from PIL import Image

IMAGE_PATH = "/home/zhaoyang/projects/VLM-Reasoning/test-ground-images-results/gpt-4o-2024-05-13/2step-object-list/pick-grounded-images/task-0/go-to-bed-1.jpg"
PTOMPT = """<image>
List all visible objects in the image. If there is no image, please type 'no image'."""

action_vlm_decoding_args = DecodingArguments(
    max_tokens=2048,
    n=1,
    temperature=0.6,
    image_detail="auto",
)
actor_vlm_model = ChatBot("gpt-4o-2024-05-13")

image = Image.open(IMAGE_PATH)
image.save("image.jpg")
messages = {"text": PTOMPT, "images": [image]}
response = actor_vlm_model.call_model(
    messages,
    decoding_args=action_vlm_decoding_args,
    return_list=False,
).strip()
# print(PTOMPT)
print(response)
