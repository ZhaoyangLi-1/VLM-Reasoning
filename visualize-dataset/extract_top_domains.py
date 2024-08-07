import json
import os
import pickle
from io import BytesIO
from base64 import b64decode
from PIL import Image
import numpy as np
from moviepy.editor import ImageSequenceClip
import random
from urllib.parse import urlparse
import argparse


dataset_folders = {
    "mmc4-ff-core": "/data3/dataset/VLM/mmc4-ff-core/domains",
    "coyo-700m": "/data3/dataset/VLM/coyo-700m/domains",
    "obelics": "/data3/dataset/VLM/obelics/domains",
}


def get_top_frequen_domains_per_dataset():
    all_sorted_dataset_domains_dict = {}
    for dataset_name, dataset_folder in dataset_folders.items():
        print(f"Processing {dataset_folder}...")
        with open(os.path.join(dataset_folder, "total.json"), "r") as f:
            dataset_domains_counts = json.load(f)
            sorted_keys = sorted(
                dataset_domains_counts.keys(), key=lambda k: dataset_domains_counts[k], reverse=True
            )
            print(f"Top 15 domains in {dataset_name}: {sorted_keys[:15]} with counts: {sorted([dataset_domains_counts[k] for k in sorted_keys[:15]])}")
            all_sorted_dataset_domains_dict[dataset_name] = sorted_keys
    output_path = "/home/zhaoyang/projects/visualize-dataset/domains/sorted_domains.pkl"
    with open(output_path, "wb") as output_file:
        pickle.dump(all_sorted_dataset_domains_dict, output_file)


def get_each_json_for_top_domains():
    with open(
        "/home/zhaoyang/projects/visualize-dataset/domains/sorted_domains.pkl", "rb"
    ) as f:
        all_sorted_dataset_domains_dict = pickle.load(f)
    for dataset_name, dataset_folder in dataset_folders.items():
        sorted_top_dataset_domains = [domain.lower() for domain in all_sorted_dataset_domains_dict[dataset_name][:15]] 
        dataset_dict = {}
        for filename in os.listdir(dataset_folder):
            if (
                "solved_domains.json" in filename
                or "total.json" in filename
                or "solved_domains.pkl" in filename
                or not filename.endswith(".json")
            ):
                continue
            print(f"Processing: {filename} in dataset: {dataset_name}")
            
            with open(os.path.join(dataset_folder, filename), "r") as f:
                dataset_domains_dict = json.load(f)
                top_donmain = set()
                top_donmain_count = 0
                domains = [domain.lower() for domain in dataset_domains_dict.keys()]
                for domain in domains:
                    if domain in sorted_top_dataset_domains:
                        top_donmain.add(domain)
                        top_donmain_count += 1
                top_doamin_length = len(top_donmain)
            dataset_dict[filename] = {"domains": list(top_donmain), "count": top_donmain_count, "unique_count": top_doamin_length}
            
        output_json_path = f"/home/zhaoyang/projects/visualize-dataset/domains/{dataset_name}_top_domains.json"
        with open(output_json_path, 'w') as json_file:
            json.dump(dataset_dict, json_file, indent=4)
        print(f"Saved dataset domain lengths to {output_json_path}")


if __name__ == "__main__":
    get_top_frequen_domains_per_dataset()
    get_each_json_for_top_domains()