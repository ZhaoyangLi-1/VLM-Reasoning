import re
import os
from utils.alfworld_utils import get_done_paths, get_path_tasks

task_log_template = """
---------------------------------------------------------task: {task_idx}---------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------------------------------------------
Step:--------------------------------------------------------------------------{step}-------------------------------------------------------------------------------
----------------------------------------
VLM Prompt:
{prompt}
----------------------------------------

----------------------------------------

Original Response:
{response}

----------------------------------------

----------------------------------------

>>> Refine Original VLM Response and Get Pure Action: {action}

----------------------------------------

-----------------------------------------------------------------

Running time: {time} seconds

Total Money: {total_money}

-----------------------------------------------------------------

-----------------------------------------------------------------

Text Observation:{obs}

-----------------------------------------------------------------
"""

html_task_log_template = """
---------------------------------------------------------<strong>task: {task_idx}</strong>---------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------------------------------------------
<strong>Step:</strong>--------------------------------------------------------------------------{step}-------------------------------------------------------------------------------
----------------------------------------

<strong>VLM Prompt:</strong>
{prompt}

----------------------------------------

----------------------------------------

<strong>Original Response:</strong>
{response}

----------------------------------------

----------------------------------------

<strong>&gt;&gt;&gt; Refine Original VLM Response and Get Pure Action:</strong> <span style="color: red;">{action}</span>

----------------------------------------

-----------------------------------------------------------------

<strong>Running time:</strong> {time} seconds

<strong>Total Money:</strong> {total_money}

-----------------------------------------------------------------

-----------------------------------------------------------------

<strong>Text Observation:</strong>{obs}

------
"""



class Logger(object):
    def __init__(self, filename):
        self.filename = filename
        f = open(self.filename, "a")
        f.close()

    def write(self, message):
        with open(self.filename, "a") as f:
            f.write(message)
            f.close()


class Global_Logger(Logger):
    def __init__(self, filename):
        super().__init__(filename)

    def read_lines(self):
        with open(self.filename, "r") as f:
            lines = f.readlines()
            f.close()
        return lines

    def skip_succeed(self, parent_folder, args):  # Changed to use self
        num_succeess = 0
        no_head = False
        lines = self.read_lines()
        done_json_paths = get_done_paths(lines)
        self.num_done = len(done_json_paths)
        if len(lines) >= 1:
            print("Continue from the last time")
            for idx, line in enumerate(lines):
                if "SUCCEED" in line and "UNSUCCEED" not in line:
                    num_succeess += 1
                if idx == 0 and "Begin task:" not in line:
                    no_head = True
        else:
            print("Start from the beginning")
            no_head = True
        task_list_path = os.path.join(parent_folder, args.task_list_path)
        json_file_list = get_path_tasks(task_list_path)
        tasks_steps = args.total_task // args.num_server
        json_file_list = sorted(
            json_file_list[
                args.begin_task : args.begin_task + tasks_steps
            ]
        )
        print(min(args.begin_task + tasks_steps, args.total_task))
        if self.num_done != 0:
            json_file_list = sorted(
                [item for item in json_file_list if item not in done_json_paths]
            )
        if no_head:
            self.write(
                f"Begin task: {args.begin_task}     End task: {args.begin_task+tasks_steps-1}      Number of Server: {args.num_server}\n"
            )
        return json_file_list


class Task_Logger(Logger):
    def __init__(self, filename, task_id):
        super().__init__(filename)
        self.task_id = task_id

    def wirte_and_get_task_log(
        self, task_idx, step, prompt, response, action, start_time, end_time, total_money, obs
    ):
        time = end_time - start_time
        task_log = task_log_template.format(
            task_idx=task_idx,
            step=step,
            prompt=prompt,
            response=response,
            action=action,
            time = time,
            total_money=total_money,
            obs=obs,
        )
        task_log = task_log[1:-1]
        html_task_log = task_log_template.format(
            task_idx=task_idx,
            step=step,
            prompt=prompt,
            response=response,
            action=action,
            time = time,
            total_money=total_money,
            obs=obs,
        )
        html_task_log = html_task_log[1:-1]
        self.write(task_log)
        return html_task_log
