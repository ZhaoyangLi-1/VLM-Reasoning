import os
import json
import re
import argparse

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def match_task_indices_to_class(result_directory, task_class_set):
    task_indices_to_class = {
        "Success Using Memory but Failed Without Memory": [],
        "Success Using Memory": [],
        "Failed Using Memory": [],
    }
    if not os.path.exists(task_class_set):
        assert False, f"Task class Set jSON file: {task_class_set} does not exist"
    if not os.path.exists(result_directory):
        assert False, f"Result directory: {result_directory} does not exist"
    with open(task_class_set, "r") as f:
        sub_exmaple_tasks_set = json.load(f)

    for filename in os.listdir(result_directory):
        if filename.endswith(".log"):
            filepath = os.path.join(result_directory, filename)
            with open(filepath, "r") as f:
                total_lines = sum(1 for line in f)
            with open(filepath, "r") as f:
                next(f)
                for line_number, line in enumerate(f, 1):
                    if line_number >= total_lines - 4:
                        break
                    index, rest = line.split(" ", 1)
                    match = re.search(r"train/[^:]*?\.json", rest)
                    if match:
                        task_path_part = match.group(0)
                        for key, task_list in sub_exmaple_tasks_set.items():
                            if task_path_part in task_list:
                                task_indices_to_class[key].append(f"task-{index}")
    return task_indices_to_class


def create_index_for_class(tasks, class_name, directory):
    index_content = f"<h2>{class_name}</h2>\n<ul>\n"
    for task in tasks:
        index_content += f'    <li><a href="{task}/index.html">{task}</a></li>\n'
        create_index_for_task(task, directory)
    index_content += "</ul>\n"
    return index_content

def create_index_for_task(task_name, result_directory):
    task_path = os.path.join(result_directory, task_name)  # Define task_path correctly
    html_directory = os.path.join(task_path, "html-files")

    # Ensure the directory exists and contains HTML files
    if not os.path.exists(html_directory):
        print(f"HTML directory {html_directory} does not exist.")
        return

    # List all HTML files in the directory
    html_files = [
        file for file in os.listdir(html_directory) if file.endswith('.html')
    ]

    # Sort the files to maintain order
    html_files = sorted(html_files, key=lambda x: int(x.split('-')[-1].split('.')[0]))

    # Start the index HTML content
    index_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Index of {os.path.basename(task_name)}</title>
</head>
<body>
    <h1>Step Index of {os.path.basename(task_name)}</h1>
    <ul>
"""

    # Loop through HTML files and add them to the index content
    for html_file in html_files:
        index_content += (
            f'        <li><a href="html-files/{html_file}">{html_file}</a></li>\n'
        )

    # Close the list and the HTML tags
    index_content += """    </ul>
</body>
</html>"""

    # Write the index content to an index.html file in the task directory
    with open(os.path.join(task_path, "index.html"), "w") as f:
        f.write(index_content)



# def main(args):
#     result_directory = os.path.join(PARENT_DIR, "result", args.result_directory) 
#     task_class_set = os.path.join(PARENT_DIR, "tasks", args.task_class_set)
#     task_indices_to_class = match_task_indices_to_class(result_directory, task_class_set)
#     master_index_content = """<!DOCTYPE html>
# <html>
# <head>
#     <title>Tasks Overview</title>
# </head>
# <body>
#     <h1>Tasks Overview</h1>
# """

#     for class_name, tasks in task_indices_to_class.items():
#         master_index_content += create_index_for_class(tasks, class_name, result_directory)

#     master_index_content += "</body>\n</html>"

#     # Write the master index content to the main directory
#     with open(os.path.join(result_directory, "index.html"), "w") as f:
#         f.write(master_index_content)

def main(args):
    base_result_directory = os.path.join(PARENT_DIR, args.result_directory)
    task_class_set = os.path.join(PARENT_DIR,"tasks", args.task_class_set)

    for entry in os.listdir(base_result_directory):
        result_directory = os.path.join(base_result_directory, entry)
        index_file_path = os.path.join(result_directory, "index.html")
        if os.path.isdir(result_directory) and not os.path.exists(index_file_path):
            print(f"Processing directory: {result_directory}")
            task_indices_to_class = match_task_indices_to_class(result_directory, task_class_set)

            master_index_content = """<!DOCTYPE html>
<html>
<head>
    <title>Tasks Overview</title>
</head>
<body>
    <h1>Tasks Overview</h1>
"""

            for class_name, tasks in task_indices_to_class.items():
                master_index_content += create_index_for_class(tasks, class_name, result_directory)

            master_index_content += "</body>\n</html>"

            # Write the master index content to the main directory
            with open(os.path.join(result_directory, "index.html"), "w") as f:
                f.write(master_index_content)
        else:
            print(f"Skipping already processed directory: {result_directory}")
    

    with open('index.html', 'w') as f:
        f.write('<!DOCTYPE html>\n<html>\n<head>\n<title>Results Overview</title>\n</head>\n<body>\n')
        f.write('<h1>Results Overview</h1>\n<ul>\n')
        
        for dir_name in os.listdir(base_result_directory):
            print(f"Processing directory: {dir_name}")
            if os.path.isdir(dir_name) and 'index.html' in os.listdir(dir_name):
                f.write(f'    <li><a href="{os.path.join(dir_name, "index.html")}">{dir_name}</a></li>\n')
        
        f.write('</ul>\n</body>\n</html>')

    print("Index file created.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-directory",
        type=str,
        default="results",
    )
    parser.add_argument("--task-class-set", type=str, default="sub_exmaple_tasks_set.json")
    args = parser.parse_args()
    main(args)
