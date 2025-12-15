import json 
from pathlib import Path 

# Read json file in batch_tasks_results_no_reward/example/result/out.json and save the data in indented format 
input_file = Path("batch_tasks_results_no_reward/example/result/out.json")
with open(input_file, "r") as f:
    data = json.load(f)

# Save the data in indented format
with open(f"{input_file.parent}/result_indented.json", "w") as f:
    json.dump(data, f, indent=4)
