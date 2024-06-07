import pandas as pd
import json
import sys
import os
import matplotlib.pyplot as plt


def compute_pick_acc():
    csv_path = "/home/zhaoyang/projects/VLM-Reasoning/grounded_pick_planning.csv"
    df = pd.read_csv(csv_path)
    df['vlm_succeed'] = df['vlm_succeed'].astype(bool)
    
    # Calculate overall accuracy
    vlm_succeed_accuracy = df['vlm_succeed'].mean()
    print(f"Overall vlm_succeed accuracy: {vlm_succeed_accuracy:.2f}")
    
    # Calculate accuracy for each task_id
    df_group_by_task = df.groupby('task_id')['vlm_succeed'].mean().reset_index()
    df_group_by_task.columns = ['task_id', 'vlm_succeed_accuracy']
    
    # Plot bar chart
    plt.figure(figsize=(12, 8))
    bars = plt.bar(df_group_by_task['task_id'], df_group_by_task['vlm_succeed_accuracy'], color='skyblue')
    plt.xlabel('Task ID')
    plt.ylabel('VLM Succeed Accuracy')
    plt.title('VLM Succeed Accuracy per Task ID')
    plt.xticks(rotation=90)
    
    # Add accuracy values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    plt.savefig("pick-acc-per-task.jpg", dpi=300)
 
    
if __name__ == "__main__":
    compute_pick_acc()