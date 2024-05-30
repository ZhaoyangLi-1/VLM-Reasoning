import pandas as pd
import os
import ast
from collections import defaultdict
import argparse
import matplotlib.pyplot as plt

CURRENT_FOLDER = os.path.dirname(os.path.abspath(__file__))


def parse_objects(obj_str):
    if obj_str == "[]":
        return []
    return ast.literal_eval(obj_str)


def draw_accuracy_per_category_bar(file_name, args, type, save_file_name):
    df = pd.read_csv(file_name)
    # Sort the data by accuracy in descending order
    df_sorted = df.sort_values(by=type, ascending=True)

    plt.figure(figsize=(8, 16))
    plt.barh(df_sorted['object_category'], df_sorted[type], color='#ED784A', height=0.6)
    plt.xlabel(f'{type.capitalize()}')
    plt.ylabel('Object Category')
    plt.title(f"{type.capitalize()} of Predicting Object Categories")
    # plt.subplots_adjust(top=0.4)
    num_categories = len(df_sorted['object_category'])
    plt.ylim(-0.6, num_categories - 0.4)
    plt.tight_layout()
    save_folder = os.path.join(CURRENT_FOLDER, args.result_csv_folder, args.vlm_result_folder, f"{args.file_type}-bar-images")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_path = os.path.join(save_folder, save_file_name)
    plt.savefig(save_path, dpi=300)
    print(f"Chart saved to {save_path}")

# true positives (TP), false positives (FP), false negatives (FN)
# Recall = TP / (TP + FN)
# Accuracy = TP / (TP + FP)
# true_objs: [Bed, SideTable, Drawer, DeskLamp, Pillow, CellPhone, Pencil], predicted_objs: ['Pillow', 'Dresser', 'SideTable', 'Bed', 'DeskLamp']
# True Positives (TP): Objects that are correctly predicted. These are the objects present in both the true and predicted lists.
# TP: Pillow, SideTable, Bed, DeskLamp (4 objects)
# False Positives (FP): Objects that are incorrectly predicted as present. These are in the predicted list but not in the true list.
# FP: Dresser (1 object)
# False Negatives (FN): Objects that are in the true list but not predicted.
# FN: Drawer, CellPhone, Pencil (3 objects)
# Recall measures the proportion of actual positives that are correctly identified by the model.
def main(args):
   # Load the CSV file
    csv_file_path = os.path.join(CURRENT_FOLDER, args.result_csv_folder, args.vlm_result_folder, args.file_name)
    df = pd.read_csv(csv_file_path)

    # Initialize dictionaries to store counts
    true_positive_counts = defaultdict(int)
    false_positive_counts = defaultdict(int)
    false_negative_counts = defaultdict(int)

    # Process each row in the DataFrame
    for _, row in df.iterrows():
        true_objects = parse_objects(row['true_objs'])
        predicted_objects = parse_objects(row['predicted_objs'])
        
        # Convert lists to sets for easier calculation
        true_set = set(true_objects)
        predicted_set = set(predicted_objects)
        
        # Calculate TPs, FPs, FNs
        true_positives = true_set.intersection(predicted_set)
        false_positives = predicted_set.difference(true_set)
        false_negatives = true_set.difference(predicted_set)
        
        # Update counts for each category
        for obj in true_positives:
            true_positive_counts[obj] += 1
        for obj in false_positives:
            false_positive_counts[obj] += 1
        for obj in false_negatives:
            false_negative_counts[obj] += 1

    # Prepare data for DataFrame
    data = []
    object_categories = set(true_positive_counts.keys()).union(false_positive_counts.keys()).union(false_negative_counts.keys())
    for category in object_categories:
        tp = true_positive_counts[category]
        fp = false_positive_counts[category]
        fn = false_negative_counts[category]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0  # Only calculate if there are predicted objects
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # This should always have a non-zero denominator as true_objects is never empty
        data.append({'object_category': category, 'Recall': recall, 'Precision': precision})

    # Create DataFrame
    metrics_df = pd.DataFrame(data)

    # Save DataFrame to CSV
    save_path_folder = os.path.join(CURRENT_FOLDER, args.result_csv_folder, args.vlm_result_folder)
    if not os.path.exists(save_path_folder):
        os.makedirs(save_path_folder)
    save_path = os.path.join(save_path_folder, f"{args.file_type}.csv")
    metrics_df.to_csv(save_path, index=False)
    
    draw_accuracy_per_category_bar(save_path, args, 'Recall', f"recall-per-category-for-predicting-objects.jpg")
    draw_accuracy_per_category_bar(save_path, args, 'Precision', f"precision-per-category-for-predicting-objects.jpg")
    

if __name__ == "__main__":
    print("Running the script for computing recall and accuracy...")
    parser = argparse.ArgumentParser(description='Compute recall and accuracy for per category across all images')
    parser.add_argument("--result_csv_folder", default="GPT4-V-result", type=str)
    parser.add_argument("--vlm_result_folder", default="no-refine", type=str)
    parser.add_argument("--file_name", default="predicted-objects-categories-list.csv", type=str)
    parser.add_argument("--file_type", default="recall-and-precision-for-predicted-objects-categories", type=str)
    args = parser.parse_args()
    main(args)