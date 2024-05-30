import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt

CURRENT_FOLDER = os.path.dirname(os.path.abspath(__file__))

def draw_accuracy_per_category_bar(df_sorted, args):
    # Create bar chart using the sorted dataframe
    plt.figure(figsize=(8, 16))
    plt.barh(df_sorted['object_category'], df_sorted['accuracy'], color='#ED784A', height=0.6)
    plt.xlabel('Accuracy')
    plt.ylabel('Object Category')
    plt.title(f"Accuracy of Top 20 Object Categories in the {args.file_type.capitalize()} Question")
    num_categories = len(df_sorted['object_category'])
    plt.ylim(-0.6, num_categories - 0.4)
    plt.tight_layout()
    save_folder = os.path.join(CURRENT_FOLDER, args.result_csv_folder, args.QA_result_folder, f"{args.file_type}-bar-images")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_path = os.path.join(save_folder, "top_20_categories_chart.jpg")
    plt.savefig(save_path, dpi=300)
    print(f"Chart saved to {save_path}")

def main(args):
    file_path = os.path.join(CURRENT_FOLDER, args.result_csv_folder, args.QA_result_folder, args.vlm_result_file_name)
    df = pd.read_csv(file_path)
    # Calculate the top 10 most frequent categories
    top_categories = df['object_category'].value_counts().nlargest(20).index
    df_top = df[df['object_category'].isin(top_categories)]
    
    # Calculate accuracy for these top categories
    grouped = df_top.groupby('object_category')
    accuracy_per_category = grouped.apply(lambda x: (x['correctness'] == 'correct').sum() / len(x))
    accuracy_df = accuracy_per_category.reset_index()
    accuracy_df.columns = ['object_category', 'accuracy']
    accuracy_df = accuracy_df.sort_values(by='accuracy', ascending=True)

    # Save the filtered accuracy data
    save_folder = os.path.join(CURRENT_FOLDER, args.result_csv_folder, args.QA_result_folder)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_path = os.path.join(save_folder, f"accuracy-top-20-category-{args.file_type}.csv")
    accuracy_df.to_csv(save_path, index=False)
    print(f"Accuracy per category has been saved to {save_path}.")

    # Draw the bar chart for top categories
    draw_accuracy_per_category_bar(accuracy_df, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_csv_folder", default="GPT4-V-result", type=str)
    parser.add_argument("--vlm_result_file_name", default="count-answer.csv", type=str)
    parser.add_argument("--QA_result_folder", default="no-refine", type=str)
    parser.add_argument("--file_type", default="counting", type=str) # existence
    args = parser.parse_args()
    main(args)

# python top_20_fre_compute_acc_and_save_bar.py --QA_result_folder edge-refine --file_type counting --vlm_result_file_name counting-answer.csv
# python top_20_fre_compute_acc_and_save_bar.py --QA_result_folder edge-refine --file_type existence --vlm_result_file_name existence-answer.csv