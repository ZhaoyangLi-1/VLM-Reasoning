import csv
import json
import os
import re

CURRENT_FOLDER = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER = os.path.join(CURRENT_FOLDER, "images/original")
# IMAGE_OBJ_INFOR_PATH = os.path.join(CURRENT_FOLDER, "image-obj.csv")

def str_to_dict(s):
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return {}
    

def read_data_from_csv(image_obj_infor_path):
    print(f"Reading images with the object information from {image_obj_infor_path} ...")
    data = []
    with open(image_obj_infor_path, 'r', newline='') as file:
        reader = csv.reader(file)
        header = next(reader)
        for row in reader:
            if row[-3].startswith('{') and row[-3].endswith('}'):
                row[-3] = str_to_dict(row[-3].replace("'", '"'))
            if row[-1].startswith('{') and row[-1].endswith('}'):
                row[-1] = str_to_dict(row[-1].replace("'", '"'))
            data.append(row)
    data_dicts = [dict(zip(header, row)) for row in data[1:]]
    return data_dicts


def process_object_string_list(input_list, is_lower_case=False):
    processed_list = []
    for input_string in input_list:
        words = re.findall('[A-Z][a-z]*', input_string)
        
        if not words:
            if is_lower_case:
                processed_list.append(input_string.lower())
            processed_list.append(input_string)
        else:
            new_string = ' '.join(words)
            if is_lower_case:
                new_string = new_string.lower()
            processed_list.append(new_string)
    return processed_list