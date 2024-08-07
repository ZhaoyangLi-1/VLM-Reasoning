import os
import pickle
import random
from io import BytesIO
from base64 import b64decode, b64encode
from PIL import Image
import numpy as np
from moviepy.editor import ImageSequenceClip
import tempfile
import json
from urllib.parse import urlparse
from tqdm import tqdm
from urllib.parse import urlparse
from collections import defaultdict

DATASETS = {
    "coyo-700m": "/data3/dataset/VLM/coyo-700m/pkls",
    "mmc4-ff-core": "/data3/dataset/VLM/mmc4-ff-core/pkls",
    "obelics": "/data3/dataset/VLM/obelics/pkls",
}
sorted_domains_path = "/home/zhaoyang/projects/visualize-dataset/domains/chosed_domain.json"

with open(sorted_domains_path, "r") as f:
    sorted_domains = json.load(f)


found_data = {}
for dataset, folder_path in DATASETS.items():
    saved_data = []
    dataset_sorted_domains = sorted_domains[dataset]
    print(f"Dataset: {dataset}")
    all_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.pkl'):
            print(f"Processing {filename}")
            file_path = os.path.join(folder_path, filename)
            # Open each pkl file
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
                all_data.extend(data)
    total_data = len(all_data)
    with tqdm(total=total_data) as pbar:
        if dataset == "coyo-700m":
            for idx, element in enumerate(all_data):
                is_choose_enough = True
                parsed_url = urlparse(element["url"])
                domain = parsed_url.netloc
                if domain not in dataset_sorted_domains:
                    # print(f"{domain} not in top domain list")
                    is_choose_enough = False
                if not is_choose_enough:
                    pbar.update(1)
                    continue
                saved_data.append((domain, element))
                pbar.update(1)

        elif dataset == "mmc4-ff-core":
            for idx, element in enumerate(all_data):
                image_info_list = element["image_info"]
                is_choose_enough = True
                domain_list =[]
                for img_idx, image_info in enumerate(image_info_list):
                    parsed_url = urlparse(image_info["raw_url"])
                    domain = parsed_url.netloc
                    domain_list.append(domain)
                    if domain not in dataset_sorted_domains:
                        is_choose_enough = False
                        break
                if not is_choose_enough:
                    pbar.update(1)
                    continue
                saved_data.append((domain_list[0], element))
                pbar.update(1)
        
        elif dataset == "obelics":
            for idx, element in enumerate(all_data):
                is_choose_enough = True
                urls = [url for url in element['image_info'] if url is not None]
                for url in urls:
                    parsed_url = urlparse(url)
                    domain = parsed_url.netloc
                    if domain not in dataset_sorted_domains:
                        is_choose_enough = False
                        break
                if not is_choose_enough:
                    pbar.update(1)
                    continue
                saved_data.append((domain, element))
                pbar.update(1)
                
            
        domain_grouped_data = defaultdict(list)
        for domain, element in saved_data:
            domain_grouped_data[domain].append(element)
        
        final_sampled_data = []
        for domain in dataset_sorted_domains:
            if domain in domain_grouped_data:
                sampled_elements = random.sample(domain_grouped_data[domain], min(15, len(domain_grouped_data[domain])))
                final_sampled_data.extend(sampled_elements)

        # If there are less than 150 elements, fill up with random elements from the existing list
        remaining_sample_size = 150 - len(final_sampled_data)
        if remaining_sample_size > 0:
            additional_sample = random.sample(saved_data, remaining_sample_size)
            final_sampled_data.extend([element for domain, element in additional_sample])
                
    output_path = f"/home/zhaoyang/projects/visualize-dataset/extracted_data/{dataset}.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(final_sampled_data, f)


file_name = "/home/zhaoyang/projects/visualize-dataset/extracted_data/coyo-700m.pkl"
with open(file_name, "rb") as f:
    breakpoint()
    data = pickle.load(f)
    print(len(data))