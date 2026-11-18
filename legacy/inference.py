import os
import re
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import subprocess
import time

def test_code(code_path):
    # code_path = f"./test.py"
    # with open(code_path, "w") as f1:
    #     f1.write(code_str)

    ans = subprocess.run(f"python {code_path}", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # return answer logs, error code
    return str(ans.stdout.decode('gbk', errors='ignore')), str(ans.stderr.decode('gbk', errors='ignore'))

    
def generate_response(model_path: str, question_path: str, output_html: str = None):
    """
    Generate a response from the fine-tuned LLM and save the mathematical model part as HTML.

    Args:
        model_path (str): Path to the fine-tuned model checkpoint directory.
        question_path (str): Path to the text file containing the question.
        output_html (str, optional): Output HTML file path. Defaults to 'model_output.html' in cwd.
    """

    # if path of text file doesnt exist throw error
    if not os.path.exists(question_path):
        raise FileNotFoundError(f"Question file not found: {question_path}")
    with open(question_path, "r", encoding="utf-8") as f:
        question_text = f.read().strip()

    prompt = f"""Below is an operations research question. Build a mathematical model and corresponding python code using `coptpy` that appropriately addresses the question.

# Question:
{question_text}

# Response:
"""
    # prompt = f"""{question_text}"""
    st_time = time.time()
    # loading model from the path
    print(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        # torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    # tokenize the inputs
    inputs = tokenizer(
        [prompt + tokenizer.eos_token],
        return_tensors="pt",
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    # generating the response from model
    print("Generating response...")
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=8192,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    # putting response into html
    html = out_to_mf_html(response)

    if output_html is None:
        output_html = os.path.join(os.getcwd(), "model_output.html")

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\nHTML file saved at: {output_html}")
    
    code = extract_python_code_from_response(response)
    if code:
        code_filename = os.path.splitext(output_html)[0] + "_code.py"
        with open(code_filename, "w", encoding="utf-8") as f:
            f.write(code)
        print(f"Python code extracted and saved at: {code_filename}")
        out_log, err_log = test_code(code_filename)
        print(out_log)
    else:
        print("No Python code block found in the response.")
    print("Time taken (in seconds):", (time.time() - st_time))
    


def out_to_mf_html(raw_text: str) -> str:
    """
    Converts model output text into styled HTML focusing on the mathematical model section.
    """

    # Extract the mathematical model section only
    if "## Mathematical Model:" in raw_text and "## Python Code" in raw_text:
        mf = raw_text.split("## Mathematical Model:")[1].split("## Python Code")[0]
    else:
        mf = raw_text  # fallback

    text = mf.replace("\\n", "\n").strip()

    # Convert Markdown-style headers to HTML
    text = re.sub(r"### (.+)", r"<h3>\1</h3>", text)
    text = re.sub(r"## (.+)", r"<h2>\1</h2>", text)
    text = re.sub(r"# (.+)", r"<h1>\1</h1>", text)

    # Convert bullet points and wrap them in <ul>
    text = re.sub(r"(?m)^\s*-\s+(.+)", r"<li>\1</li>", text)
    text = re.sub(r"((?:<li>.*?</li>\n?)+)", r"<ul>\1</ul>", text)

    # Add paragraph tags for standalone lines
    text = re.sub(r"(?m)^(?!<h\d>|<ul>|<li>|</li>|</ul>)([^<\n]+)$", r"<p>\1</p>", text)

    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Mathematical Model</title>
        <script type="text/javascript" async
            src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
        </script>
        <style>
            body {{
                font-family: 'Cambria', serif;
                margin: 2cm;
                line-height: 1.6;
                background-color: #f9fafc;
                color: #111;
            }}
            h1, h2, h3 {{
                color: #003366;
                margin-top: 1em;
            }}
            p {{
                font-size: 14pt;
                text-align: justify;
            }}
            li {{
                font-size: 14pt;
                margin-left: 1em;
            }}
            ul {{
                margin-top: 0.5em;
                margin-bottom: 0.5em;
            }}
        </style>
    </head>
    <body>
    {text}
    </body>
    </html>
    """

    return html_template

import re

def extract_python_code_from_response(response: str) -> str:
    """
    Extracts Python code enclosed in ```python ... ``` blocks from a model response.

    Args:
        response (str): The full text output from the model.

    Returns:
        str: The extracted Python code as a single string.
             If multiple code blocks are found, they are concatenated with separators.
             Returns an empty string if no code blocks are found.
    """
    # Find all code blocks starting with ```python and ending with ```
    code_blocks = re.findall(r"```python(.*?)```", response, re.DOTALL | re.IGNORECASE)

    if not code_blocks:
        return ""

    # Clean and join multiple blocks
    cleaned_code = '''\n\n# --- Next Block ---\n\n'''.join(block.strip() for block in code_blocks)
    return cleaned_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint (e.g., output/checkpoint-4500)")
    parser.add_argument("--question_file", type=str, required=True, help="Path to text file containing the OR question")
    parser.add_argument("--output_html", type=str, help="Optional: Path to save output HTML file")

    args = parser.parse_args()
    generate_response(args.model_path, args.question_file, args.output_html)
