# import csv
# import json
# import wordninja
# import os

# CURRENT_FILE_PATH = os.path.dirname(os.path.abspath(__file__))

# def split_camel_case(text):
#     words = wordninja.split(text)
#     return ''.join(word.capitalize() if i == 0 else word.lower() for i, word in enumerate(words))

# def process_object_bboxes(json_str):
#     # 替换单引号为双引号以符合 JSON 格式
#     json_str = json_str.replace("'", '"')
#     obj_dict = json.loads(json_str)
#     new_dict = {split_camel_case(key): value for key, value in obj_dict.items()}
#     return json.dumps(new_dict, ensure_ascii=False)

# def update_object_infor(object_infor, object_bboxes):
#     # 替换单引号为双引号以符合 JSON 格式
#     object_infor = object_infor.replace("'", '"')
#     object_bboxes = object_bboxes.replace("'", '"')
    
#     object_infor_dict = json.loads(object_infor)
#     object_bboxes_dict = json.loads(object_bboxes)
    
#     object_bboxes_keys_lower = {key.lower() for key in object_bboxes_dict.keys()}
    
#     # 确保键是字符串类型
#     new_object_infor = {key: value for key, value in object_infor_dict.items() if key.lower() in object_bboxes_keys_lower}
    
#     return json.dumps(new_object_infor, ensure_ascii=False)

# # 路径应根据你的实际文件位置进行调整
# csv_file_path = os.path.join(CURRENT_FILE_PATH, 'edge-refine-image-obj.csv')
# output_file_path = os.path.join(CURRENT_FILE_PATH,'edge-refine-image-obj-revise.csv')

# # 读取CSV文件
# rows = []
# with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
#     reader = csv.DictReader(file)
#     for row in reader:
#         processed_bboxes = process_object_bboxes(row['object_bboxes'])
#         updated_infor = update_object_infor(row['object_infor'], processed_bboxes)
#         row['object_bboxes'] = processed_bboxes
#         row['object_infor'] = updated_infor
#         rows.append(row)

# # 将更新后的数据写回CSV
# with open(output_file_path, mode='w', newline='', encoding='utf-8') as file:
#     writer = csv.DictWriter(file, fieldnames=rows[0].keys())
#     writer.writeheader()
#     writer.writerows(rows)


