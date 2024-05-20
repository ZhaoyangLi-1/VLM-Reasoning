import argparse
from baselines.base_direct_act import test_tasks as test_tasks_base
from use_memory.use_memory_act import test_tasks as test_tasks_use_memory

def run_aflworld(args):
    if "baseline" in args.mode:
        test_tasks_base(args)
    elif "use_memory" in args.mode:
        test_tasks_use_memory(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm_model", default="gpt-4-vision-preview", type=str)
    parser.add_argument("--use-4bit", default=True, type=bool)
    parser.add_argument("--use-8bit", default=False, type=bool)
    parser.add_argument("--llm_model", default="gpt-4-1106-preview", type=str)
    parser.add_argument(
        "--save_path",
        # default="one-image-direct-action-selection",
        default="test-existence-use-memory-act-2",
        type=str,
    )
    parser.add_argument(
        "--task_list_path",
        type=str,
        default="tasks/sub_exmaple_tasks_set.json",
    )
    parser.add_argument(
        "--env_url",
        type=str,
        default=3000,
        # required=True
    )
    parser.add_argument(
        "--is_ins_seg",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--begin_task",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--num_server",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--total_task",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=1,
    )    
    parser.add_argument(
        "--mode",
        type=str,
        default="use_memory", # baseline
    )
    parser.add_argument(
        "--max_step",
        type=int,
        default=11,
    )
    parser.add_argument("--QA_Mode", default="existence", type=str)
    parser.add_argument("--refine_type", default="no-refine", type=str)
    parser.add_argument("--result_csv_folder", default="GPT4-V-result", type=str)
    args = parser.parse_args()
    run_aflworld(args)
    print("Finished!")