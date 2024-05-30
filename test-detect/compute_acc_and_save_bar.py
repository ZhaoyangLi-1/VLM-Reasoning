import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt

CURRENT_FOLDER = os.path.dirname(os.path.abspath(__file__))

def draw_accuracy_per_category_bar(file_name, args):
    df = pd.read_csv(file_name)
    # Sort the data by accuracy in descending order
    df_sorted = df.sort_values(by='accuracy', ascending=True)

    plt.figure(figsize=(8, 16))
    plt.barh(df_sorted['object_category'], df_sorted['accuracy'], color='#ED784A', height=0.6)
    plt.xlabel('Accuracy')
    plt.ylabel('Object Category')
    plt.title(f"Accuracy of Object Categories in the {args.file_type.capitalize()} Question")
    # plt.subplots_adjust(top=0.4)
    num_categories = len(df_sorted['object_category'])
    plt.ylim(-0.6, num_categories - 0.4)
    plt.tight_layout()
    save_folder = os.path.join(CURRENT_FOLDER, args.result_csv_folder, args.QA_result_folder, f"{args.file_type}-bar-images")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_path = os.path.join(save_folder, "all_categories_chart.jpg")
    plt.savefig(save_path, dpi=300)
    print(f"Chart saved to {save_path}")

def main(args):
    file_path = os.path.join(CURRENT_FOLDER, args.result_csv_folder, args.QA_result_folder, args.vlm_result_file_name)
    df = pd.read_csv(file_path)

    
    grouped = df.groupby('object_category')

    save_folder = os.path.join(CURRENT_FOLDER, args.result_csv_folder, "existence-bar-images")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_path = os.path.join(save_folder, "object-bar.jpg")
    print(f"Saving the plot to {save_path}")
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    accuracy_per_category = grouped.apply(lambda x: (x['correctness'] == 'correct').sum() / len(x))
    
    # Calculate overall accuracy
    overall_accuracy = (df['correctness'] == 'correct').sum() / len(df)
    print(f"Overall accuracy across all categories: {overall_accuracy:.2f}")
    
    accuracy_df = accuracy_per_category.reset_index()
    accuracy_df.columns = ['object_category', 'accuracy']
    save_folder = os.path.join(CURRENT_FOLDER, args.result_csv_folder, args.QA_result_folder)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_path = os.path.join(save_folder, f"accuracy-per-category-{args.file_type}.csv")
    accuracy_df.to_csv(save_path, index=False)
    print(f"Accuracy per category has been saved to {save_path}.")
    draw_accuracy_per_category_bar(save_path, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_csv_folder", default="GPT4-V-result-balance", type=str)
    parser.add_argument("--vlm_result_file_name", default="existence-answer.csv", type=str)
    parser.add_argument("--QA_result_folder", default="edge-refine", type=str)
    parser.add_argument("--file_type", default="existence", type=str) # existence
    args = parser.parse_args()
    main(args)
    
# python compute_acc_and_save_bar.py --result_csv_folder GPT4-V-result-balance --QA_result_folder edge-refine --file_type counting --vlm_result_file_name counting-answer.csv
# python compute_acc_and_save_bar.py --result_csv_folder GPT4-V-result-balance --QA_result_folder edge-refine --file_type existence --vlm_result_file_name existence-answer.csv
