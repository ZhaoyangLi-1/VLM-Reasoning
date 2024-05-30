import os
import sys
import argparse
import csv
import re
from PIL import Image
import ast
import pandas as pd
import matplotlib.pyplot as plt
import random
import statsmodels.api as sm

os.environ["AGI_ROOT"] = "/home/zhaoyang/projects/neural-reasoning"
sys.path.append(os.path.join(os.environ["AGI_ROOT"]))
from agi.utils.chatbot_utils import DecodingArguments, ChatBot
import wordninja

CURRENT_FOLDER = os.path.dirname(os.path.abspath(__file__))
PRPMPT_PATH = os.path.join(CURRENT_FOLDER, "prompts/hisoty_existence_question.txt")
ALFOWORLD_RESULT_PATH = os.path.join(CURRENT_FOLDER, "gpt-aflworld-history-infor.csv")


def split_and_capitalize(input_str):
    # Split the string into words
    words = wordninja.split(input_str)
    # Capitalize each word and join
    return "".join(word.capitalize() for word in words)


def camel_case_to_lower(s):
    s = re.sub(r"(?<!^)(?=[A-Z])", " ", s)
    return s.lower().replace(" ", "")


def camel_case_to_words(s):
    s = re.sub(r"(?<!^)(?=[A-Z])", " ", s)
    return s.lower()


def get_all_objs(data_for_all_images):
    objects_for_all_images = set()
    for _, row in data_for_all_images.iterrows():
        object_count_dic = ast.literal_eval(row["object_count_infor"])
        if len(object_count_dic) == 0:
            continue
        objects_for_all_images.update(object_count_dic.keys())
    return list(objects_for_all_images)


def generate_no_existed_data(object_infor, all_objects):
    objects_in_image = list(object_infor.keys())
    num_objects_in_image = len(objects_in_image)
    available_objs_candidates = list(set(all_objects) - set(objects_in_image))
    selected_objs_candidates = random.sample(
        available_objs_candidates,
        min(len(available_objs_candidates), num_objects_in_image),
    )
    final_objs_dict = {}
    for obj in selected_objs_candidates:
        final_objs_dict[obj] = 0
    return final_objs_dict


def write_data_to_csv(filename, data, mode="a"):
    with open(filename, mode, newline="") as file:
        writer = csv.writer(file)
        writer.writerow(data)
        print(f"Added row: {data}")


def extract_number(s):
    # Updated to handle non-integer strings gracefully
    match = re.search(r"\d+", s)
    return int(match.group()) if match else None


def process_response(response, counts, QA_Mode):
    if "count" in QA_Mode:
        counts_hat = extract_number(response)
        correctness = "correct" if counts_hat == counts else "incorrect"
    else:
        if "Yes" in response and counts > 0 or "No" in response and counts == 0:
            correctness = "correct"
        else:
            correctness = "incorrect"
        counts_hat = 1 if "Yes" in response else 0
    return counts_hat, correctness


# def balance_and_add_task_related_objects(
#     all_objects_in_image, all_objects, task_related_objects
# ):
#     task_related_and_all_objects_in_image = list(
#         set(all_objects_in_image + task_related_objects)
#     )
#     task_related_and_all_objects_in_image_dic = {}
#     for item in task_related_and_all_objects_in_image:
#         if item in all_objects_in_image:
#             task_related_and_all_objects_in_image_dic[item] = 1
#         # Otherwise, check if it's in task_related_objects and not in all_objects_in_image
#         elif item in task_related_objects:
#             task_related_and_all_objects_in_image_dic[item] = 0
#     # Count the number of 1s and 0s
#     count_1 = sum(
#         1 for value in task_related_and_all_objects_in_image_dic.values() if value == 1
#     )
#     count_0 = sum(
#         1 for value in task_related_and_all_objects_in_image_dic.values() if value == 0
#     )
#     balance_needed = count_1 - count_0
#     if balance_needed > 0:
#         potential_items_to_add = [
#             item
#             for item in all_objects
#             if item not in task_related_and_all_objects_in_image_dic
#         ]
#         items_to_add = random.sample(
#             potential_items_to_add, min(balance_needed, len(potential_items_to_add))
#         )
#         for item in items_to_add:
#             task_related_and_all_objects_in_image_dic[item] = 0
#     return task_related_and_all_objects_in_image_dic

def balance_and_add_objects(all_objects_in_image, all_objects):
    objects_in_image_dic = {item: 1 for item in all_objects_in_image}

    count_in_image = len(objects_in_image_dic)
    
    balance_needed = count_in_image % 2

    if balance_needed != 0:
        potential_items_to_add = [
            item for item in all_objects if item not in objects_in_image_dic
        ]
        items_to_add = random.sample(
            potential_items_to_add, min(balance_needed, len(potential_items_to_add))
        )
        for item in items_to_add:
            objects_in_image_dic[item] = 0

    return objects_in_image_dic


def run_test_with_vlm(args):
    prompt_template = open(PRPMPT_PATH, "r").read()
    print(f"Prompt Template:\n{prompt_template}")

    action_vlm_decoding_args = DecodingArguments(
        max_tokens=2048,
        n=1,
        temperature=0.6,
        image_detail="auto",
    )

    csv_save_folder = os.path.join(CURRENT_FOLDER, args.result_csv_folder)
    if not os.path.exists(csv_save_folder):
        os.makedirs(csv_save_folder)

    csv_save_path = os.path.join(csv_save_folder, "existence-answer.csv")
    if "gpt" in args.vlm_model:
        actor_vlm_model = ChatBot(args.vlm_model)
    else:
        actor_vlm_model = ChatBot(
            args.vlm_model, "http://localhost:41000", use_cpp=False
        )
    df = pd.read_csv(ALFOWORLD_RESULT_PATH)
    all_objects = get_all_objs(df)
    header = [
        "task_id",
        "task_desc",
        "ins_image_path",
        "object_category",
        "is_task_related",
        "true_counts",
        "predicted_counts",
        "correctness",
        "history_num",
        "history",
        "object_count_infor",
    ]
    write_data_to_csv(csv_save_path, header, mode="w")
    for _, row in df.iterrows():
        task_id = row["task_id"]
        task_desc = row["task_desc"]
        history = row["history"]
        object_count_infor = ast.literal_eval(row["object_count_infor"])

        if len(object_count_infor) == 0:
            continue

        original_image_path = row["original_image_path"]
        ins_image_path = row["ins_image_path"]
        task_related_objects = ast.literal_eval(row["task_related_objects"])
        task_related_objects = [
            split_and_capitalize(obj) for obj in task_related_objects
        ]
        history_num = row["history_num"]

        all_objects_in_image = list(object_count_infor.keys())
        # all_objects_for_detect = balance_and_add_task_related_objects(
        #     all_objects_in_image, all_objects, task_related_objects
        # )
        all_objects_for_detect = balance_and_add_objects(all_objects_in_image, all_objects)
        image = Image.open(original_image_path)

        for object_name, counts in all_objects_for_detect.items():
            is_task_related = 1 if object_name in task_related_objects else 0
            prompt = prompt_template.format(
                history=history, object_name=camel_case_to_words(object_name)
            )
            messages = {
                "text": prompt,
                "images": [image],
            }
            response = actor_vlm_model.call_model(
                messages, decoding_args=action_vlm_decoding_args, return_list=False
            ).strip()
            counts_hat, correctness = process_response(response, counts, args.QA_Mode)

            # answer.append(['task_id', 'task_desc', 'ins_image_path', 'object_category', 'is_task_related', 'true_counts', 'predicted_counts', 'correctness', 'history_num', 'history', 'object_count_infor'])
            raw_data = [
                task_id,
                task_desc,
                ins_image_path,
                object_name,
                is_task_related,
                counts,
                counts_hat,
                correctness,
                history_num,
                history,
                object_count_infor,
            ]
            print(
                f"Task ID: {task_id}, Task Description: {task_desc}, Object: {object_name}, True Counts: {counts}, Predicted Counts: {counts_hat}, Correctness: {correctness}"
            )
            write_data_to_csv(csv_save_path, raw_data)


def draw_objects_bar(df, args):
    # Print and plot the count of each object_category
    category_counts = df["object_category"].value_counts()
    print(f"Count of each object_category: {category_counts}")
    # print()

    # Plotting the horizontal bar graph for object_category counts
    plt.figure(
        figsize=(10, 16)
    )  # Adjust the figure size as needed for better visualization
    category_counts.sort_values().plot(
        kind="barh", color="#ED784A"
    )  # Using 'barh' for horizontal bars
    plt.title("Count of Each Object Categories in the Existence Question")
    plt.xlabel("Count")
    plt.ylabel("Object Category")
    plt.tight_layout()  # Adjusts plot to ensure everything fits without overlap

    # Define the save folder based on the current setup, ensure this matches your file structure
    CURRENT_FOLDER = os.getcwd()
    save_folder = os.path.join(
        CURRENT_FOLDER, args.result_csv_folder, "existence-bar-images"
    )
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_path = os.path.join(save_folder, "object-bar.jpg")
    print(f"Chart saved to {save_path}")
    plt.savefig(save_path, dpi=300)
    plt.close()


def compute_acc(df):
    # Compute the accuracy per history number
    grouped_history = df.groupby("history_num")
    accuracy_grouped_per_history_num = grouped_history.apply(
        lambda x: (x["correctness"] == "correct").sum() / len(x)
    )
    accuracy_per_history_num_df = accuracy_grouped_per_history_num.reset_index()
    accuracy_per_history_num_df.columns = ["history_num", "accuracy"]

    # Compute the accuracy per category within each history number
    grouped_category = df.groupby(["history_num", "object_category"])
    accuracy_grouped_per_category = grouped_category.apply(
        lambda x: (x["correctness"] == "correct").sum() / len(x)
    )
    accuracy_per_category_df = accuracy_grouped_per_category.reset_index()
    accuracy_per_category_df.columns = ["history_num", "object_category", "accuracy"]

    # Compute the average accuracy for each object_category for each history_num
    average_accuracy_per_object_df = (
        accuracy_per_category_df.groupby(["object_category", "history_num"])
        .mean()
        .reset_index()
    )
    average_accuracy_per_object_df.columns = [
        "object_category",
        "history_num",
        "average_accuracy",
    ]

    return (
        accuracy_per_history_num_df,
        accuracy_per_category_df,
        average_accuracy_per_object_df,
    )


def draw_accuracy_per_history_num_bar(df, args):
    # Sort the data by accuracy in descending order
    # df_sorted = df.sort_values(by='accuracy', ascending=True)

    plt.figure(figsize=(8, 8))
    plt.plot(df["history_num"], df["accuracy"], color="#ED784A")
    plt.xlabel("History Number")
    plt.ylabel("Accuracy")
    plt.title(
        f"Accuracy of Object Categories in the Existence Question vs. History Number"
    )
    # plt.subplots_adjust(top=0.4)
    num_categories = len(df["history_num"])
    plt.xlim(-0.6, num_categories - 0.4)
    plt.tight_layout()
    save_folder = os.path.join(
        CURRENT_FOLDER, args.result_csv_folder, "existence-bar-images"
    )
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_path = os.path.join(save_folder, "accuracy-for-history-number.jpg")
    plt.savefig(save_path, dpi=300)
    print(f"Chart saved to {save_path}")


def draw_accuracy_per_object_for_all_history_num_bar(df, args):
    grouped_object = df.groupby("object_category")

    num_categories = len(grouped_object)
    categories = list(grouped_object)

    half = num_categories // 2

    fig1, axes1 = plt.subplots(
        nrows=(half + 3) // 4, ncols=4, figsize=(12, 2 * ((half + 3) // 4))
    )
    axes1 = axes1.flatten()

    for ax, (name, group) in zip(axes1, categories[:half]):
        ax.plot(group["history_num"], group["average_accuracy"], marker="o")
        ax.set_title(f"Category: {name}")
        ax.set_xlabel("History Number")
        ax.set_ylabel("Average Accuracy")
        ax.set_ylim(0, 1.05)

    for ax in axes1[half:]:
        ax.set_visible(False)

    plt.tight_layout()

    save_folder = os.path.join(
        CURRENT_FOLDER, args.result_csv_folder, "existence-bar-images"
    )
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    save_path1 = os.path.join(
        save_folder, "average-accuracy-for-history-number-per-object1.jpg"
    )
    plt.savefig(save_path1, dpi=300)
    print(f"Chart saved to {save_path1}")

    plt.close(fig1)

    fig2, axes2 = plt.subplots(
        nrows=((num_categories - half + 3) // 4),
        ncols=4,
        figsize=(12, 2 * ((num_categories - half + 3) // 4)),
    )
    axes2 = axes2.flatten()

    for ax, (name, group) in zip(axes2, categories[half:num_categories]):
        ax.plot(group["history_num"], group["average_accuracy"], marker="o")
        ax.set_title(f"Category: {name}")
        ax.set_xlabel("History Number")
        ax.set_ylabel("Average Accuracy")
        ax.set_ylim(0, 1.05)

    for ax in axes2[num_categories - half :]:
        ax.set_visible(False)

    plt.tight_layout()

    save_path2 = os.path.join(
        save_folder, "average-accuracy-for-history-number-per-object2.jpg"
    )
    plt.savefig(save_path2, dpi=300)
    print(f"Chart saved to {save_path2}")

    plt.close(fig2)


def draw_action_bar():
    import pandas as pd
    import matplotlib.pyplot as plt
    import os

    # Load data
    original_inor_df = pd.read_csv(
        "/home/zhaoyang/projects/VLM-Reasoning/test-detect/gpt-aflworld-history-infor.csv"
    )
    original_inor_df.iloc[:, 3] = (
        original_inor_df.iloc[:, 3]
        .astype(str)
        .str.replace(r"\d+", "", regex=True)
        .str.split("\n")
        .str[0]
    )
    action_counts = original_inor_df.iloc[:, 3].value_counts()

    # Create larger figure to better accommodate labels
    plt.figure(figsize=(14, 18))  # Adjust width here if necessary

    # Plot data
    action_counts.sort_values().plot(kind="barh", color="#ED784A")

    # Set titles and labels
    plt.title("Count of Actions in 30 Tasks")
    plt.xlabel("Count")
    plt.ylabel("Action")

    # Adjust layout
    # plt.subplots_adjust(left=0.35, right=0.9)
    plt.tight_layout()
    # Save plot
    save_folder = os.path.join(
        CURRENT_FOLDER, args.result_csv_folder, "existence-bar-images"
    )
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_path = os.path.join(save_folder, "action-bar.jpg")
    print(f"Saving the plot to {save_path}")
    plt.savefig(save_path, dpi=300)
    plt.close()


def compute_and_acc_for_each_object_in_each_history_num(args, df, history_num):
    df_filtered = df[df["history_num"] == history_num]

    grouped_object = df_filtered.groupby("object_category")
    accuracy_grouped_per_object = grouped_object.apply(
        lambda x: (x["correctness"] == "correct").sum() / len(x)
    )

    accuracy_per_object_df = accuracy_grouped_per_object.reset_index()
    accuracy_per_object_df.columns = ["object_category", "accuracy"]

    fig, ax = plt.subplots(
        figsize=(12, 20)
    )  # Adjust the figure size as needed for better visualization
    accuracy_per_object_df.sort_values(by="accuracy", ascending=False).plot(
        kind="barh", x="object_category", y="accuracy", color="#ED784A", ax=ax
    )
    plt.title(f"Accuracy of Each Object Category for History Number {history_num}")
    plt.xlabel("Accuracy")
    plt.ylabel("Object Category")

    # Adjust the spacing to prevent label overlap
    plt.subplots_adjust(left=0.3, right=0.95, top=0.95, bottom=0.05)

    # Increase the distance between the bars
    ax.yaxis.set_tick_params(pad=10)

    # Define the save folder based on the current setup, ensure this matches your file structure
    save_folder = os.path.join(
        CURRENT_FOLDER, args.result_csv_folder, "history-num-object-bar-images"
    )
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_path = os.path.join(
        save_folder, f"object-bar-history-history-num-{history_num}.jpg"
    )
    print(f"Chart saved to {save_path}")
    plt.savefig(save_path, dpi=300)
    plt.close()

    return accuracy_per_object_df


def compute_acc_of_history_num_with_each_task(df, task_id):

    def draw_accuracy_per_history_num_bar(df, task_id):
        plt.figure(figsize=(8, 8))
        plt.plot(df["history_num"], df["accuracy"], color="#ED784A")
        plt.xlabel("History Number")
        plt.ylabel("Accuracy")
        plt.title(
            f"Accuracy of Object Categories in the Existence Question vs. History Number in Task {task_id}"
        )
        # plt.subplots_adjust(top=0.4)
        num_categories = len(df["history_num"])
        plt.xlim(-0.6, num_categories - 0.4)
        plt.tight_layout()
        save_folder = os.path.join(
            CURRENT_FOLDER,
            args.result_csv_folder,
            "each-task-accuracy-vs-history-number",
        )
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        save_path = os.path.join(
            save_folder, f"task-{task_id}-accuracy-for-history-number.jpg"
        )
        plt.savefig(save_path, dpi=300)
        print(f"Chart saved to {save_path}")

    def calculate_accuracy(group):
        total = len(group)
        correct = (group["correctness"] == "correct").sum()
        accuracy = correct / total
        return accuracy

    df_task_with_id = df[df["task_id"] == task_id]
    accuracy_by_group = (
        df_task_with_id.groupby("history_num")
        .apply(calculate_accuracy)
        .reset_index(name="accuracy")
    )
    draw_accuracy_per_history_num_bar(accuracy_by_group, task_id)


def compute_acc_for_each_object_in_specific_diff_history_num(
    args, df, history_num_1, history_num_2
):

    def plot_accuracy(accuracy_grouped_per_object, title, save_name, color="#ED784A"):
        plt.figure(figsize=(14, 6))
        accuracy_grouped_per_object.set_index("object_category")[
            "accuracy"
        ].sort_values().plot(kind="barh", color=color)
        plt.xlabel("Accuracy")
        plt.ylabel("Object Category")
        plt.title(title)

        save_folder = os.path.join(
            CURRENT_FOLDER, args.result_csv_folder, "history-num-object-bar-images"
        )
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        unique_save_path = os.path.join(save_folder, save_name)
        plt.savefig(unique_save_path, dpi=300)
        plt.close()
        print(f"Chart saved to {unique_save_path}")

    def plot_common_accuracy(
        accuracy_1, accuracy_2, history_num1, history_num2, title, save_name
    ):
        plt.figure(figsize=(22, 18))
        bar_height = 0.4

        merged_df = accuracy_1.merge(accuracy_2, on="object_category", suffixes=("_1", "_2"))
        merged_df = merged_df.sort_values(by="accuracy_1")
        
        
        r1 = range(len(merged_df))
        r2 = [x + bar_height for x in r1]

        plt.barh(
            r1,
            merged_df["accuracy_1"],
            color="#ED784A",
            height=bar_height,
            edgecolor="grey",
            label=f"History Number {history_num1}",
        )
        plt.barh(
            r2,
            merged_df["accuracy_2"],
            color="#5DA5DA",
            height=bar_height,
            edgecolor="grey",
            label=f"History Number {history_num2}",
        )

        plt.xlabel("Accuracy")
        plt.ylabel("Object Category")
        plt.title(title)
        plt.yticks(
            [r + bar_height / 2 for r in range(len(merged_df))],
            merged_df["object_category"],
        )
        plt.legend()

        save_folder = os.path.join(
            CURRENT_FOLDER, args.result_csv_folder, "history-num-object-bar-images"
        )
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        unique_save_path = os.path.join(save_folder, save_name)
        plt.savefig(unique_save_path, dpi=300)
        plt.close()
        print(f"Chart saved to {unique_save_path}")

    def calculate_accuracy(df, unique_objects):
        grouped_object = df[df["object_category"].isin(unique_objects)].groupby(
            "object_category"
        )
        accuracy_grouped_per_object = grouped_object.apply(
            lambda x: (x["correctness"] == "correct").sum() / len(x)
        ).reset_index()
        accuracy_grouped_per_object.columns = ["object_category", "accuracy"]
        return accuracy_grouped_per_object

    df_filtered_1 = df[df["history_num"] == history_num_1]
    df_filtered_2 = df[df["history_num"] == history_num_2]

    unique_objects_1 = set(df_filtered_1["object_category"].unique())
    unique_objects_2 = set(df_filtered_2["object_category"].unique())
    common_objects = unique_objects_1 & unique_objects_2

    unique_to_history_num_1 = unique_objects_1 - unique_objects_2
    unique_to_history_num_2 = unique_objects_2 - unique_objects_1

    accuracy_grouped_per_object_1 = calculate_accuracy(
        df_filtered_1, unique_to_history_num_1
    )
    accuracy_grouped_per_object_2 = calculate_accuracy(
        df_filtered_2, unique_to_history_num_2
    )
    accuracy_common_objects_1 = calculate_accuracy(df_filtered_1, common_objects)
    accuracy_common_objects_2 = calculate_accuracy(df_filtered_2, common_objects)

    print(
        f"Unique objects for History Number {history_num_1}: {unique_to_history_num_1}"
    )
    print(
        f"Unique objects for History Number {history_num_2}: {unique_to_history_num_2}"
    )

    plot_accuracy(
        accuracy_grouped_per_object_1,
        f"Unique Object Categories for History Number {history_num_1}",
        f"unique-object-bar-history-num-{history_num_1}.jpg",
    )
    plot_accuracy(
        accuracy_grouped_per_object_2,
        f"Unique Object Categories for History Number {history_num_2}",
        f"unique-object-bar-history-num-{history_num_2}.jpg",
    )
    plot_common_accuracy(
        accuracy_common_objects_1,
        accuracy_common_objects_2,
        history_num_1,
        history_num_2,
        f"Common Object Categories for History Numbers {history_num_1} and {history_num_2}",
        f"common-object-bar-history-num-{history_num_1}-and-{history_num_2}.jpg",
    )

    merged_df = pd.merge(
        accuracy_common_objects_1,
        accuracy_common_objects_2,
        on="object_category",
        suffixes=("_1", "_2"),
    )
    result = merged_df[merged_df["accuracy_2"] > merged_df["accuracy_1"]][
        "object_category"
    ]
    len_result = len(result)
    len_inverse_result_with_no_equal = len( merged_df[merged_df["accuracy_2"] < merged_df["accuracy_1"]]["object_category"])
    print(f"Accuracy of {history_num_2} > {history_num_1}: {len_result}")
    print(f"Accuracy of {history_num_2} < {history_num_1}: {len_inverse_result_with_no_equal}")
    # Filter df for history_num_2 and result
    filtered_df = df[
        (df["history_num"] == history_num_2) | (df["history_num"] == history_num_1) & (df["object_category"].isin(result))
    ]

    # Add the accuracy columns to the filtered dataframe
    filtered_df = pd.merge(
        filtered_df,
        accuracy_common_objects_1[["object_category", "accuracy"]],
        on="object_category",
        how="left",
    )
    filtered_df = pd.merge(
        filtered_df,
        accuracy_common_objects_2[["object_category", "accuracy"]],
        on="object_category",
        how="left",
        suffixes=(f"_{history_num_1}", f"_{history_num_2}"),
    )

    save_folder = os.path.join(
        CURRENT_FOLDER, args.result_csv_folder, "high-acc-object-history-num"
    )
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_path = os.path.join(
        save_folder,
        f"high-acc-object-history-num-between-{history_num_1}-{history_num_2}.csv",
    )
    filtered_df.to_csv(save_path, index=False)
    print(f"Data saved to {save_path}")


def compute_acc_for_each_object_in_task(df, task_id):

    def draw_accuracy_per_history_num_bar(df, task_id, object_category):
        plt.figure(figsize=(14, 8))
        plt.plot(
            df["history_num"],
            df["accuracy"],
            marker="o",
            linestyle="-",
            color="#ED784A",
        )
        plt.xlabel("History Number")
        plt.ylabel("Accuracy")
        plt.title(
            f"Accuracy of Object Categories in the Existence Question vs. History Number in Task {task_id} with Object Category {object_category}"
        )
        plt.xlim(df["history_num"].min() - 0.5, df["history_num"].max() + 0.5)
        plt.ylim(0, 1.05)
        plt.tight_layout()
        save_folder = os.path.join(
            CURRENT_FOLDER,
            args.result_csv_folder,
            "task-object-accuracy-vs-history-number",
            f"task-{task_id}-object-accuracy-vs-history-number",
        )
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        save_path = os.path.join(
            save_folder, f"task-{task_id}-accuracy-{object_category}-history-number.jpg"
        )
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Chart saved to {save_path}")

    def calculate_accuracy(group):
        total = len(group)
        correct = (group["correctness"] == "correct").sum()
        accuracy = correct / total
        return accuracy

    # Filter the dataframe for the given task_id
    df_task_with_id = df[df["task_id"] == task_id]

    # Group by object_category and history_num, then calculate accuracy
    grouped = (
        df_task_with_id.groupby(["object_category", "history_num"])
        .apply(lambda group: calculate_accuracy(group))
        .reset_index(name="accuracy")
    )

    # Get unique object categories
    object_categories = grouped["object_category"].unique()

    # Plot accuracy vs history_num for each object category
    for object_category in object_categories:
        df_object = grouped[grouped["object_category"] == object_category]
        draw_accuracy_per_history_num_bar(df_object, task_id, object_category)


def clean_result(df):
    # df = df[~df["object_category"].str.contains("Cd") & ~df["object_category"].str.contains("Safe")
    #         & ~df["object_category"].str.contains("CoolPot") & ~df["object_category"].str.contains("CleanTomato")
    #         & ~df["object_category"].str.contains("HotApple") & ~df["object_category"].str.contains("CoolMug")
    #         & ~df["object_category"].str.contains("Kettle2") & ~df["object_category"].str.contains("Kettle1")
    #         & ~df["object_category"].str.contains("CleanPlate") & ~df["object_category"].str.contains("CoolApple")
    #         & ~df["object_category"].str.contains("PotOrPan") & ~df["object_category"].str.contains("CoolTomato")
    #         & ~df["object_category"].str.contains("CoolPan")]
    
    df = df.groupby("object_category").head(10)
    return df


def main(args):
    if args.is_eval:
        run_test_with_vlm(args)
    csv_save_folder = os.path.join(CURRENT_FOLDER, args.result_csv_folder)
    csv_save_path = os.path.join(csv_save_folder, "existence-answer.csv")
    print(f"Reading data from {csv_save_path}")
    df = pd.read_csv(csv_save_path)
    df = clean_result(df)
    
    draw_objects_bar(df, args)
    # df = df.groupby('object_category').filter(lambda x: len(x) >= 10).groupby('object_category').head(10)
    acc_history_num_df, acc_category_df, avg_acc_object_df = compute_acc(df)

    # Optionally, save these DataFrames
    acc_history_num_df.to_csv(os.path.join(csv_save_folder, 'accuracy_per_history_num.csv'), index=False)
    acc_category_df.to_csv(os.path.join(csv_save_folder, 'accuracy_per_category.csv'), index=False)
    avg_acc_object_df.to_csv(os.path.join(csv_save_folder, 'average_accuracy_per_object.csv'), index=False)

    draw_accuracy_per_history_num_bar(acc_history_num_df, args)
    draw_accuracy_per_object_for_all_history_num_bar(avg_acc_object_df, args)
    draw_action_bar()

    task_ids = df['task_id'].unique()
    for tak_id in task_ids:
        compute_acc_of_history_num_with_each_task(df, tak_id)

    for task_id in task_ids:
        compute_acc_for_each_object_in_task(df, task_id)

    compute_acc_for_each_object_in_specific_diff_history_num(
        args, df.groupby("object_category").head(20), 1, 2
    )
    compute_acc_for_each_object_in_specific_diff_history_num(
        args, df.groupby("object_category").head(20), 4, 6
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_csv_folder",
        default="GPT4-V-result-balance/history-edge-refine-without-task-related-objects",
        type=str,
    )
    parser.add_argument("--vlm_model", default="gpt-4-1106-vision-preview", type=str)
    parser.add_argument("--QA_Mode", default="existence", type=str)
    parser.add_argument("--is_eval", default=False, type=bool)
    args = parser.parse_args()
    main(args)
