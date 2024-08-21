import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from scipy.linalg import sqrtm
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from itertools import combinations
from sklearn.metrics import mutual_info_score
import spacy

# Ensure NLTK's tokenizer is available
nltk.download('punkt')

# Load SpaCy model for NLP tasks
nlp = spacy.load("en_core_web_sm")

class Evaluator:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.smooth_fn = SmoothingFunction().method1

    def compute_lexical_diversity(self, captions):
        all_words = []
        for caption in captions:
            words = nltk.word_tokenize(caption.lower())
            all_words.extend(words)
        total_words = len(all_words)
        unique_words = len(set(all_words))
        lexical_diversity = unique_words / total_words if total_words > 0 else 0
        return lexical_diversity

    def compute_self_bleu(self, captions):
        bleu_scores = []
        for cap1, cap2 in combinations(captions, 2):
            ref = [cap1.split()]
            hyp = cap2.split()
            bleu_score = sentence_bleu(ref, hyp, smoothing_function=self.smooth_fn)
            bleu_scores.append(bleu_score)
        return sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0

    def compute_clip_score(self, image, captions):
        inputs = self.processor(text=captions, images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds
        clip_scores = torch.cosine_similarity(image_embeds, text_embeds).cpu().numpy()
        mean_clip_score = clip_scores.mean()
        return mean_clip_score
    
    def compute_mutual_information(self, image, captions):
        inputs_image = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs_image).cpu().numpy().flatten()
        
        inputs_text = self.processor(text=captions, return_tensors="pt", padding=True)
        with torch.no_grad():
            caption_features = self.model.get_text_features(**inputs_text).cpu().numpy()
        
        mi_scores = []
        for caption_feature in caption_features:
            caption_feature_flat = caption_feature.flatten()
            mi_score = mutual_info_score(image_features, caption_feature_flat)
            mi_scores.append(mi_score)
        
        return np.mean(mi_scores)

    def extract_caption_objects(self, caption):
        doc = nlp(caption)
        return [chunk.text.lower() for chunk in doc.noun_chunks]

    def compute_object_hallucination_rate(self, detected_objects, captions):
        hallucination_rates = []
        for caption in captions:
            caption_objects = self.extract_caption_objects(caption)
            hallucinated_objects = [obj for obj in caption_objects if obj not in detected_objects]
            hallucination_rate = len(hallucinated_objects) / len(caption_objects) if caption_objects else 0
            hallucination_rates.append(hallucination_rate)
        return np.mean(hallucination_rates)
    
    def compute_visual_textual_consistency(self, detected_objects, captions):
        consistency_scores = []
        for caption in captions:
            caption_objects = self.extract_caption_objects(caption)
            overlap = set(caption_objects).intersection(set(detected_objects))
            consistency_score = len(overlap) / len(set(caption_objects).union(set(detected_objects))) if caption_objects else 0
            consistency_scores.append(consistency_score)
        return np.mean(consistency_scores)

    def run(self, image, captions, detected_objects):
        lexical_diversity = self.compute_lexical_diversity(captions)
        self_bleu = self.compute_self_bleu(captions)
        clip_score = self.compute_clip_score(image, captions)
        mi_score = self.compute_mutual_information(image, captions)
        hallucination_rate = self.compute_object_hallucination_rate(detected_objects, captions)
        vtc_score = self.compute_visual_textual_consistency(detected_objects, captions)
        
        return {
            "lexical_diversity": lexical_diversity,
            "self_bleu": self_bleu,
            "clip_score": clip_score,
            "mutual_information": mi_score,
            "object_hallucination_rate": hallucination_rate,
            "visual_textual_consistency": vtc_score
        }

# Example usage:
if __name__ == "__main__":
    evaluator = Evaluator()

    # Example data
    image = Image.open("example_image.jpg")
    captions = [
        "A person riding a bicycle in the park.",
        "Someone is biking through a green park.",
        "A cyclist is enjoying a ride in the open field."
    ]
    
    # Example detected objects (from ground truth or object detection model)
    detected_objects = ["person", "bicycle", "park", "tree"]

    # Run evaluations
    results = evaluator.run(image, captions, detected_objects)
    
    for metric, score in results.items():
        print(f"{metric}: {score:.4f}")
