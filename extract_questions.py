import os
import json

# Path to your NL4OPT JSONL file
input_file = "nl4opt_test.jsonl"

# Folder where the question files will be saved
output_folder = "questions"
os.makedirs(output_folder, exist_ok=True)

# Read line by line and save each question into its own text file
with open(input_file, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if not line.strip():
            continue
        obj = json.loads(line)
        question = obj.get("question", "").strip()
        
        # File name pattern
        filename = f"q_{i}.txt"
        filepath = os.path.join(output_folder, filename)
        
        # Write question text to file
        with open(filepath, "w", encoding="utf-8") as qf:
            qf.write(question)
        
print(f"Saved {i + 1} question files in '{output_folder}/'")
