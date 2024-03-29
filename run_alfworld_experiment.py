import argparse
from baselines.base_direct_act import test_scenes

def run_aflworld(args):
    if "baseline" in args.mode:
        test_scenes(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm_model", default="gpt-4-vision-preview", type=str)
    parser.add_argument(
        "--save_path",
        default="one-image-direct-action-selection",
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
        "--begin_scene",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--num_server",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--total_scene",
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
        default="baseline",
    )
    args = parser.parse_args()
    run_aflworld(args)
    print("Finished!")