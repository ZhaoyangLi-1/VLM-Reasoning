import argparse
from use_memory.use_memory_act import test_tasks as test_tasks_use_memory


def run_aflworld(args):
    test_tasks_use_memory(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm_model", default="gpt-4-1106-vision-preview", type=str)
    # parser.add_argument("--vlm_model", default="llava-1.6-vicuna-13b", type=str)
    parser.add_argument("--use-4bit", default=True, type=bool)
    parser.add_argument("--use-8bit", default=False, type=bool)
    parser.add_argument("--llm_model", default="gpt-4-1106-preview", type=str)
    parser.add_argument(
        "--save_path",
        default="one-image-direct-action-selection-add-object-list",
        # default="llava-v1.6-7b-existence-use-memory-act-2",
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
        "--max_step",
        type=int,
        default=20,
    )
    parser.add_argument("--is_generate_object_list", default=False, type=bool)
    parser.add_argument("--QA_Mode", default="existence", type=str)
    parser.add_argument("--refine_type", default="no-refine", type=str)
    parser.add_argument("--result_csv_folder", default="GPT4-V-result", type=str)
    args = parser.parse_args()
    run_aflworld(args)
    print("Finished!")
