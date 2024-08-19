from agi.utils.chatbot_utils import DecodingArguments, ChatBot
import argparse
import os
import json


VQAV2_FOLDER = "/data3/dataset/VLM/vqav2"
GQA_FOLDER = "/data3/dataset/VLM/GQA"


def create_question(folder_path, dataset, chatbot, decoding_args):
    with open(
        "/home/zhaoyang/projects/VLM-Reasoning/create-vlm-questions/prompts/cluster_question.txt",
        "r",
    ) as f:
        cluster_basic_prompt = f.read()
    if "vqav2" in dataset:
        image_path = "/data3/dataset/VLM/coco/train2014"
        questions_vqav2_path = os.path.join(
            folder_path, "v2_OpenEnded_mscoco_train2014_questions.json"
        )
        answers_vqav2_path = os.path.join(
            folder_path, "v2_mscoco_train2014_annotations.json"
        )

        with open(questions_vqav2_path, "r") as f:
            questions = json.load(f)["questions"]
        with open(answers_vqav2_path, "r") as f:
            annotations = json.load(f)["annotations"]

        # for idx in range(questions):
        #     question = questions[idx]['question']
        # image_id = questions[idx]['image_id']
        # image_file_path = os.path.join(image_path, f"COCO_train2014_{image_id:012d}.jpg")
        question_list = [q["question"] for q in questions][:100]
        questions_str = str(question_list)
        print(f"Length of questions: {len(questions)}")
        prompt = cluster_basic_prompt.format(questions=questions_str)
        # print(f"Prompt: {prompt}")
        messages = {"text": prompt}
        response = chatbot.call_model(
            messages, decoding_args=decoding_args, return_list=False
        ).strip()
        print(f"Response: {response}")


def main(args):
    chatbot = ChatBot(args.vlm_model)
    decoding_args = DecodingArguments(
        max_tokens=128000,
        n=1,
        temperature=0.3,
        image_detail="auto",
    )
    create_question(
        "/data3/dataset/VLM/vqav2", "vqav2", chatbot, decoding_args
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm_model", type=str, default="gpt-4-turbo-2024-04-09")
    args = parser.parse_args()
    main(args)
