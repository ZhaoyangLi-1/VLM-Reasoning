from flask import Flask, render_template, request, redirect, url_for
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

CURRENT_FOLDER = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

min_samples = 50


DATASETS = {
    "new-vflan": "/data3/dataset/VLM/new-vflan/",
    "text-flan": "/data3/dataset/VLM/new-vflan/text_flan_1m.pkl",
    "coyo-700m": os.path.join(CURRENT_FOLDER, "extracted_data/coyo-700m.pkl"),
    "mmc4-ff-core": os.path.join(CURRENT_FOLDER, "extracted_data/mmc4-ff-core.pkl"),
    "obelics":  os.path.join(CURRENT_FOLDER, "extracted_data/obelics.pkl")
}

solved_domains_path = os.path.join(CURRENT_FOLDER, "domains/chosed_domain.json")

@app.route("/")
def index():
    return render_template("index.html", datasets=DATASETS)


@app.route("/visualize", methods=["POST"])
def visualize():
    dataset_name = request.form["dataset"]
    if dataset_name in DATASETS:
        return redirect(url_for("show_dataset", dataset=dataset_name))
    return redirect(url_for("index"))


@app.route("/dataset/<dataset>")
def show_dataset(dataset):
    folder_path = DATASETS[dataset]

    with open(solved_domains_path, "rb") as f:
        domain_data = json.load(f)
    samples = []

    if dataset == "new-vflan":
        samples = process_new_vflan(folder_path)
    elif dataset == "text-flan":
        samples = process_text_flan(folder_path)
    elif dataset == "coyo-700m":
        samples = process_coyo_700m(folder_path, domain_data["coyo-700m"])
    elif dataset == "mmc4-ff-core":
        samples = process_mmc4_ff_core(folder_path, domain_data["mmc4-ff-core"])
    elif dataset == "obelics":
        samples = process_obelics(folder_path, domain_data["obelics"])

    return render_template("dataset.html", samples=samples, dataset=dataset, zip=zip)


def process_obelics(folder_path, domain_list):
    samples = []
    with open(folder_path, "rb") as f:
        all_data = pickle.load(f)

    for idx, element in enumerate(all_data):
        metadata = json.loads(element["metadata"])
        general_metadata = json.loads(element["general_metadata"])
        
        urls = [url for url in element['image_info'] if url is not None]
        parsed_url = urlparse(urls[0])
        domain = parsed_url.netloc
        domain_idx = domain_list.index(domain)
        domain = f"Top {domain_idx+1} frequency domain: {domain}"
        
        sample = {
            "domain": domain,
            "idx": idx,
            "metadata": metadata,
            "general_metadata": general_metadata,
            "contents": [],
        }

        # Iterate through images and texts and interleave them
        for image_base64_str, text in zip(element["images"], element["texts"]):
            if image_base64_str is not None:
                image = Image.open(BytesIO(b64decode(image_base64_str)))
                image = image.convert("RGB")
                buffered = BytesIO()
                image.save(buffered, format="JPEG")
                img_str = b64encode(buffered.getvalue()).decode("utf-8")
                sample["contents"].append({"type": "image", "content": img_str})
            if text is not None:
                sample["contents"].append({"type": "text", "content": text})

        samples.append(sample)

    return samples



def process_new_vflan(folder_path):
    all_files_and_folders = os.listdir(folder_path)
    files = [
        f
        for f in all_files_and_folders
        if os.path.isfile(os.path.join(folder_path, f)) and f != "tmp"
    ]
    target_size = (320, 568)
    frame_rate = 3
    samples = []

    for file in files:
        file_path = os.path.join(folder_path, file)

        with open(file_path, "rb") as f:
            data = pickle.load(f)
            data_length = len(data)
            sample_size = min(min_samples, data_length)
            sampled_indices = random.sample(range(data_length), sample_size)

            for idx in sampled_indices:
                sample = {
                    "idx": idx,
                    "question": data[idx]["question"],
                    "answer": data[idx]["answer"],
                    "images": [],
                    "video": None, 
                }

                image_base64_str_list = data[idx]["image"]
                images = []
                for img_idx, image_base64_str in enumerate(image_base64_str_list):
                    image = Image.open(BytesIO(b64decode(image_base64_str)))
                    image = image.convert("RGB")
                    image = image.resize(target_size)
                    buffered = BytesIO()
                    image.save(buffered, format="JPEG")
                    img_str = b64encode(buffered.getvalue()).decode("utf-8")

                    sample["images"].append({"img_idx": img_idx, "img_str": img_str})
                    images.append(np.array(image))

                # Create video if there are multiple images
                if len(images) > 1:
                    clip = ImageSequenceClip(images, fps=frame_rate)
                    with tempfile.NamedTemporaryFile(
                        delete=True, suffix=".mp4"
                    ) as temp_video:
                        clip.write_videofile(
                            temp_video.name, codec="libx264", audio=False, logger=None
                        )
                        temp_video.seek(0)
                        video_data = temp_video.read()
                        sample["video"] = b64encode(video_data).decode("utf-8")

                samples.append(sample)

    return samples



def process_text_flan(folder_path):
    samples = []
    with open(folder_path, "rb") as f:
        data = pickle.load(f)
        data_length = len(data)
        sample_size = min(min_samples, data_length)
        sampled_indices = random.sample(range(data_length), sample_size)

        for idx in sampled_indices:
            sample = {
                "idx": idx,
                "question": data[idx]["question"],
                "answer": data[idx]["answer"],
                "images": [],
            }
            samples.append(sample)
    return samples


def process_coyo_700m(folder_path, domain_list):
    samples = []
    with open(folder_path, "rb") as f:
        all_data = pickle.load(f)
    for idx, element in enumerate(all_data):
        parsed_url = urlparse(element["url"])
        domain = parsed_url.netloc
        domain_idx = domain_list.index(domain)
        domain = f"Top {domain_idx+1} frequency domain: {domain}"
        sample = {
            "domain": domain,
            "idx": idx,
            "text": element["text"],
            "images": [],
        }
        image_base64_str = element["image"]
        image = Image.open(BytesIO(b64decode(image_base64_str)))
        image = image.convert("RGB")
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = b64encode(buffered.getvalue()).decode("utf-8")

        sample["images"].append({"img_idx": 0, "img_str": img_str})
        samples.append(sample)
    return samples


def process_mmc4_ff_core(folder_path, domain_list):
    samples = []
    with open(folder_path, "rb") as f:
        all_data = pickle.load(f)

    for idx, element in enumerate(all_data):
        text_list = element["text_list"]
        formatted_text_list = "<br>".join(
            [f"Text {i}: {row}" for i, row in enumerate(text_list)]
        )

        similarity_matrix = element["similarity_matrix"]
        formatted_similarity_matrix = "<br>".join(
            [
                f"Image Index {i}: [{' '.join([f'{sim:.3f}' for sim in row])}]"
                for i, row in enumerate(similarity_matrix)
            ]
        )

        sample = {
            "idx": idx,
            "text_list": formatted_text_list,
            "similarity_matrix": formatted_similarity_matrix,
            "images": [],
        }
        image_info_list = element["image_info"]
        for img_idx, image_info in enumerate(image_info_list):
            parsed_url = urlparse(image_info["raw_url"])
            domain = parsed_url.netloc
            domain_idx = domain_list.index(domain)
            domain = f"Top {domain_idx+1} frequency domain: {domain}"
            matched_text_index = image_info["matched_text_index"]
            matched_text = text_list[
                matched_text_index
            ]
            matched_sim = image_info["matched_sim"]
            face_detections = image_info["face_detections"]
            image_base64_str = image_info["image_base64"]


            image = Image.open(BytesIO(b64decode(image_base64_str)))
            image = image.convert("RGB")
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            img_str = b64encode(buffered.getvalue()).decode("utf-8")

            sample["images"].append(
                {
                    "domain": domain,
                    "img_idx": img_idx,
                    "img_str": img_str,
                    "matched_text": matched_text,
                    "matched_sim": matched_sim,
                    "face_detections": face_detections,
                    "matched_text_index": matched_text_index,
                }
            )

        samples.append(sample)
    return samples


if __name__ == "__main__":
    app.run(debug=True)
