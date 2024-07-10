import pandas as pd
import json
import sys
import os
import matplotlib.pyplot as plt
import argparse

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# OLD_PICK_CSV_PATH = os.path.join(CURRENT_DIR, "grounded_pick_planning.csv") 
# OLD_PUT_CSV_PATH = os.path.join(CURRENT_DIR, "grounded_put_planning.csv")
WRONG_TASK_CSV_PATH = os.path.join(CURRENT_DIR, "wrong_task.csv")

# NEW_PICK_CSV_PATH = os.path.join(CURRENT_DIR, "grounded_pick_planning_new.csv") 
# NEW_PUT_CSV_PATH = os.path.join(CURRENT_DIR, "grounded_put_planning_new.csv")


def delete_wrong_tasks():
    # Load the CSV files into DataFrames
    old_pick_df = pd.read_csv(OLD_PICK_CSV_PATH)
    old_put_df = pd.read_csv(OLD_PUT_CSV_PATH)
    wrong_task_df = pd.read_csv(WRONG_TASK_CSV_PATH)
    
    # Get the list of wrong task IDs
    wrong_task_ids = wrong_task_df['task_id'].values
    
    # Create new DataFrames without rows with wrong task IDs
    new_pick_df = old_pick_df[~old_pick_df['task_id'].isin(wrong_task_ids)]
    new_put_df = old_put_df[~old_put_df['task_id'].isin(wrong_task_ids)]
    
    # Rename the column 'vlm_succeed' to 'action_succeed' in new_pick_df
    if 'vlm_succeed' in new_pick_df.columns:
        new_pick_df = new_pick_df.rename(columns={'vlm_succeed': 'action_succeed'})
    
    new_put_df['old_task_id'] = new_put_df['task_id']
    new_put_df['task_id'] = range(len(new_put_df))
    task_id_map = dict(zip(new_put_df['old_task_id'], new_put_df['task_id']))
    
    new_pick_df['old_task_id'] = new_pick_df['task_id']
    new_pick_df['task_id'] = new_pick_df['task_id'].map(task_id_map)
    
    # Save the cleaned DataFrames back to CSV (optional)
    new_pick_df.to_csv(NEW_PICK_CSV_PATH, index=False)
    new_put_df.to_csv(NEW_PUT_CSV_PATH, index=False)
    
    # Return the new DataFrames
    return new_pick_df, new_put_df
    

def compute_pick_acc_and_draw(df, save_path):
    df['action_succeed'] = df['action_succeed'].astype(bool)
    # Calculate overall accuracy
    action_succeed_accuracy = df['action_succeed'].mean()
    print(f"Overall action_succeed accuracy: {action_succeed_accuracy:.2f}")
    
    # Calculate accuracy for each task_id
    df_group_by_task = df.groupby('task_id')['action_succeed'].mean().reset_index()
    df_group_by_task.columns = ['task_id', 'action_succeed_accuracy']
    
    # Plot bar chart
    plt.figure(figsize=(12, 8))
    bars = plt.bar(df_group_by_task['task_id'], df_group_by_task['action_succeed_accuracy'], color='skyblue')
    plt.xlabel('Task ID')
    plt.ylabel('VLM Succeed Accuracy')
    plt.title('VLM Succeed Accuracy per Task ID')
    plt.xticks(rotation=90)
    
    # Add accuracy values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300)
    
def compute_put_acc_and_draw(df, save_path):
    df['is_vlm_response_pu'] = df['is_vlm_response_pu'].astype(bool)
    # Calculate overall accuracy
    action_succeed_accuracy = df['is_vlm_response_pu'].mean()
    print(f"Overall action_succeed accuracy: {action_succeed_accuracy:.2f}")
    
    # Calculate accuracy for each task_id
    df_group_by_task = df.groupby('task_id')['is_vlm_response_pu'].mean().reset_index()
    df_group_by_task.columns = ['task_id', 'action_succeed_accuracy']
    
    # Plot bar chart
    plt.figure(figsize=(12, 8))
    bars = plt.bar(df_group_by_task['task_id'], df_group_by_task['action_succeed_accuracy'], color='skyblue')
    plt.xlabel('Task ID')
    plt.ylabel('VLM Succeed Accuracy')
    plt.title('VLM Succeed Accuracy per Task ID')
    plt.xticks(rotation=90)
    
    # Add accuracy values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300)



def main(args):
    method_folder = os.path.join(CURRENT_DIR, "test-ground-images-results", args.vlm_model, args.method_type)
    
    global OLD_PICK_CSV_PATH, OLD_PUT_CSV_PATH, WRONG_TASK_CSV_PATH, NEW_PICK_CSV_PATH, NEW_PUT_CSV_PATH
    
    OLD_PICK_CSV_PATH = os.path.join(method_folder,  "grounded_pick_planning.csv")
    OLD_PUT_CSV_PATH = os.path.join(method_folder, "grounded_put_planning.csv")
    NEW_PICK_CSV_PATH = os.path.join(method_folder, "grounded_pick_planning_new.csv") 
    NEW_PUT_CSV_PATH = os.path.join(method_folder, "grounded_put_planning_new.csv")
    
    
    new_pick_df, new_put_df = delete_wrong_tasks()
        
    compute_pick_acc_and_draw(new_pick_df, os.path.join(method_folder, "pick-acc-per-task.jpg"))
    compute_put_acc_and_draw(new_put_df, os.path.join(method_folder, "put-acc-per-task.jpg"))
    print("Done!")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm_model", type=str, default="gpt-4o-2024-05-13")
    parser.add_argument("--method_type", type=str, default="1step-object-list")
    args = parser.parse_args()
    main(args)