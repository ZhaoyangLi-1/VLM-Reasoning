import requests
from PIL import Image, ImageDraw, ImageFont
import torch
from utils import read_data_from_csv, process_object_string_list
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

CURRENT_FOLDER = os.path.dirname(os.path.abspath(__file__))


# def add_photo_description_to_list(input_list):
#     return ["a photo of " + element for element in input_list]
      
def adjust_text_position(text_boxes, x, y, text_width, text_height):
    move_distance = 30  # Adjust as needed
    has_overlap = True
    while has_overlap:
        has_overlap = False
        for bx, by, bw, bh in text_boxes:
            if (x < bx + bw and x + text_width > bx and y < by + bh and y + text_height > by):
                y += move_distance
                has_overlap = True
                break
    return x, y


def draw_bbox_on_image(image, result, obj, original_image_path):
     # Create a drawing context
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    text_boxes = []
        
    predicted_boxes, scores, labels = result["boxes"], result["scores"], result["labels"]
    for predicted_box, score, label in zip(predicted_boxes, scores, labels):
        # box_center_x = (predicted_box[0] + predicted_box[2]) / 2
        # box_center_y = (predicted_box[1] + predicted_box[3]) / 2

        text = f"{obj[label]}: {score:.2f}"
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width, text_height = (text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1])

        # # Center the text horizontally and position it above the box vertically
        # x = box_center_x - text_width / 2
        # y = box_center_y - text_height / 2
        
        # # Ensure text does not go outside of the image boundaries
        # x = max(x, 0)
        # y = max(y, 0)
        
        x, y = predicted_box[0], predicted_box[1] - text_height - 10
        # Check for overlaps and adjust if necessary
        x, y = adjust_text_position(text_boxes, x, y, text_width, text_height)
        draw.text((x, y), text, fill="black", font=font)
        text_boxes.append((x, y, text_width, text_height))

        # Draw the rectangle after text positioning to avoid overlap issues
        draw.rectangle([predicted_box[0], predicted_box[1], predicted_box[2], predicted_box[3]], outline="red", width=2)
    
    image_save_folder = os.path.join(CURRENT_FOLDER, args.result_save_folder, args.refine_type, "annotated_images")
    if not os.path.exists(image_save_folder):
        os.makedirs(image_save_folder)
    annotated_image_path = os.path.join(
        image_save_folder, f"{os.path.basename(original_image_path)}_annotated.jpg"
    )
    image.save(annotated_image_path)
    print(f"Annotated image saved to {annotated_image_path}")


def format_predicted_bboxs(result, objs):
    prediect_boxs = {}
    objs = [obj.replace(" ", "") for obj in objs]
    predicted_boxes, scores, labels = result["boxes"], result["scores"], result["labels"]
    for predicted_box, score, label in zip(predicted_boxes, scores, labels):
        obj = objs[label]
        if obj not in prediect_boxs:
            prediect_boxs[obj] = ([], [])
        prediect_boxs[obj][0].append(predicted_box)
        prediect_boxs[obj][1].append(score)
    for obj in prediect_boxs:
        boxes_tensor = torch.stack(prediect_boxs[obj][0], dim=0)
        scores_tensor = torch.tensor(prediect_boxs[obj][1])
        prediect_boxs[obj] = (boxes_tensor, scores_tensor)
    return prediect_boxs


def format_true_bboxs(true_boxes):
    for obj in true_boxes:
        true_boxes[obj] = torch.tensor(true_boxes[obj])
    return true_boxes


def delete_object_infor_without_in_truth(object_infor, true_object_infor):
    for obj_name in list(object_infor.keys()):
        if obj_name.lower() not in true_object_infor:
            del object_infor[obj_name]
    return object_infor
    
        
def main(args):
    processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained(
        "google/owlv2-base-patch16-ensemble"
    )
    image_obj_infor_path = os.path.join(CURRENT_FOLDER, args.image_obj_infor_path)
    data = read_data_from_csv(image_obj_infor_path)
    # OWL_ViT2_result = []
    predictions_bbox_save_folder = os.path.join(CURRENT_FOLDER, args.result_save_folder, args.refine_type, "detection-results")
    ground_truth_bbox_save_folder = os.path.join(CURRENT_FOLDER, args.result_save_folder, args.refine_type, "ground-truth")
    if not os.path.exists(predictions_bbox_save_folder):
        os.makedirs(predictions_bbox_save_folder)
    if not os.path.exists(ground_truth_bbox_save_folder):
        os.makedirs(ground_truth_bbox_save_folder)
    for idx, item in enumerate(data):
        if idx + 1 == 4:
            a=1
        true_boxes_for_all_objs = format_true_bboxs(item["object_bboxes"])
        true_object_infor = list(item["object_bboxes"].keys())
        objs_infor = item["object_infor"]
        objs_infor = delete_object_infor_without_in_truth(objs_infor, true_object_infor)
        if len(objs_infor) == 0:
            continue
        objs_infor = list(objs_infor.keys())
        objs_infor = [process_object_string_list(objs_infor, is_lower_case=True)]
        original_image_path = item["original_image_path"]
        image_base_name = os.path.basename(original_image_path).split(".")[0]
        image = Image.open(original_image_path)
        inputs = processor(text=objs_infor, images=image, return_tensors="pt")
        outputs = model(**inputs)
        target_sizes = torch.Tensor([image.size[::-1]])
        results = processor.post_process_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=0.1
        )
        obj = objs_infor[0]
        result = results[0]
        draw_bbox_on_image(image, result, obj, original_image_path.split(".")[0])
        prediected_boxs_for_all_objs = format_predicted_bboxs(result, obj)
        with open(os.path.join(predictions_bbox_save_folder, f"{image_base_name}.txt"), "w") as f:
            for obj in prediected_boxs_for_all_objs:
                boxes, scores = prediected_boxs_for_all_objs[obj]
                for box, score in zip(boxes, scores):
                    f.write(f"{obj} {score:.6f} {int(box[0])} {int(box[1])} {int(box[2])} {int(box[3])}\n")
            f.close()
        
        with open(os.path.join(ground_truth_bbox_save_folder, f"{image_base_name}.txt"), "w") as f:
            for obj in true_boxes_for_all_objs:
                boxes = true_boxes_for_all_objs[obj]
                for box in boxes:
                    f.write(f"{obj} {box[0]} {box[1]} {box[2]} {box[3]}\n")
            f.close()
        
        print(f"Processed {idx + 1}/{len(data)} images")
    print("Finished processing all images")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Object Detection with OWL-ViT2")
    parser.add_argument("--refine_type", default="edge-refine", type=str)
    parser.add_argument("--result_save_folder", default="OWL-ViT2-result", type=str)
    parser.add_argument("--image_obj_infor_path", default="edge-refine-image-obj.csv", type=str)
    args = parser.parse_args()
    main(args)
