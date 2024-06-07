import os
import argparse
import random
import pickle

# TASK_TYPES = [
#     "pick_and_place_simple",
#     "look_at_obj_in_light",
#     "pick_clean_then_place_in_recep",
#     "pick_heat_then_place_in_recep",
#     "pick_cool_then_place_in_recep",
#     "pick_two_obj_and_place",
# ]

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

TASK_TYPES = [
    "pick_and_place_simple"
]


def get_json_files(data_path, base_path):
    num = 0
    train_data_list = []
    for filepath, _, filenames in os.walk(data_path):
        for filename in filenames:
            json_path = os.path.join(filepath, filename)
            if any(task_type in filepath for task_type in TASK_TYPES) and filename.endswith("traj_data.json"):
                num += 1
                relative_path = os.path.relpath(json_path, base_path)  # Compute relative path
                print(f"Task {num}: {relative_path}")
                if "Sliced" in relative_path:
                    continue
                train_data_list.append(relative_path)

    return train_data_list

def main(args):
    ALFWORLD_DATA = os.getenv("ALFWORLD_DATA")
    if ALFWORLD_DATA is None:
        raise ValueError("Environment variable 'ALFWORLD_DATA' is not set.")
    alfread_json_path = os.path.join(ALFWORLD_DATA, args.alfread_json_path)
    print(f"Current Path is : {alfread_json_path}")
    json_file_list = get_json_files(alfread_json_path, ALFWORLD_DATA)  # Pass ALFWORLD_DATA as the base path
    # print("Search Done")
    # with open("ori_task_json.pkl", 'wb') as f:
    #     pickle.dump(json_file_list, f)
    # Ensure we only sample if we have enough files to meet the requested sample size
    sample_size = min(30, len(json_file_list))
    random_selected_samples = sorted(random.sample(json_file_list, sample_size))
    print("Save Done")
    # Store the list in a file for access by other scripts
    save_path = os.path.join(CURRENT_PATH, args.save_path)
    with open(args.save_path, 'wb') as f:
        for record in random_selected_samples:
            f.write(f"{record}\n".encode())


if __name__ == "__main__":
    print("Running alfworld_memory_cot_new.py")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alfread-json-path",
        type=str,
        default="train",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="simple_pick_up_task_json.txt",
    )
    args = parser.parse_args()
    main(args)
