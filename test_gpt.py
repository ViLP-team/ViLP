import os
import base64
import requests
import json
import numpy as np
import pandas as pd
from io import BytesIO
from PIL import Image
from huggingface_hub import hf_hub_download
import argparse

from eval_utils import compare_outputs, normalize_output  

# ================== ARGUMENT PARSING ================== #
def parse_arguments():
    parser = argparse.ArgumentParser(description="Run OpenAI GPT-4 Vision Evaluation.")
    parser.add_argument(
        '--api_key', 
        required=True, 
        help="Your OpenAI API key."
    )
    parser.add_argument(
        '--openai_model', 
        default="gpt-4o-mini-2024-07-18",
        help="OpenAI model to use."
    )
    parser.add_argument(
        '--without_fact', 
        action='store_true',
        default=False,
        help="Whether to include the fact sentence in the question."
    )
    return parser.parse_args()

args = parse_arguments()

# ================== API CONFIGURATION ================== #
API_URL = "https://api.openai.com/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {args.api_key}",
    "Content-Type": "application/json",
}

# ================== HUGGING FACE LOAD ================== #
parquet_path = hf_hub_download(
    repo_id="ViLP/ViLP",     
    filename="ViLP.parquet", 
    repo_type="dataset",
    use_auth_token=True
)
df = pd.read_parquet(parquet_path)
print('Columns in dataset:', df.columns)
print('Total rows:', len(df))
# The following columns are: 
# [ 'question_id', 'question', 'image1', 'answer1', 'image2', 'answer2', 'image3', 'answer3' ]

# ================== IMAGE PREPARATION ================== #
def process_image(image_bytes, size=(1024, 1024)):
    """
    Process an image given as raw bytes, optionally resize/convert to JPEG,
    then re-encode it to base64 for sending to the API.
    """
    with Image.open(BytesIO(image_bytes)) as img:
        img = img.convert("RGB")
        img = img.resize(size, Image.LANCZOS)

        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        buffer.seek(0)
        encoded_jpeg = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return encoded_jpeg

# ================== REQUEST PREPARATION ================== #
def prepare_requests(df, without_fact):
    """
    Convert each row of the dataframe into multiple requests 
    (3 images -> 3 requests). Returns:
      - requests_batch: a list of request payloads
      - gt_results: the ground-truth answers for each image
      - question_list: the question for each image (so we can log later)
    """
    requests_batch = []
    gt_results = []
    question_list = []  # Keep track of question strings for each request

    for i in range(len(df)):
        row = df.iloc[i]
        question = row['question']
        if without_fact:
            question = question.split('.')[1].strip()
        image_cols = ['image1', 'image2', 'image3']
        answer_cols = ['answer1', 'answer2', 'answer3']

        for img_col, ans_col in zip(image_cols, answer_cols):
            image_bytes = row[img_col]
            gt_answer = row[ans_col]
            if pd.isna(image_bytes) or pd.isna(gt_answer):
                continue

            processed_img_base64 = process_image(image_bytes)

            gt_results.append(gt_answer)


            question_list.append(question)

            # Build request for OpenAI API
            request_payload = {
                "model": args.openai_model,  # or whichever GPT model you need
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Please answer with one word: {question}"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{processed_img_base64}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 300
            }
            requests_batch.append(request_payload)

    return requests_batch, gt_results, question_list

requests_batch, gt_results, question_list = prepare_requests(df, args.without_fact)

# ================== API REQUEST EXECUTION ================== #
batch_size = len(df)  # or tweak as needed
our_results = []
count = 0

# We will collect detailed entries to store in JSON later
inference_records = []

print("\nStart inference on ViLP dataset:")
print('---')
for i in range(0, len(requests_batch), batch_size):
    batch = requests_batch[i : i + batch_size]
    for request_payload in batch:
        response = requests.post(API_URL, headers=headers, json=request_payload)
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content'].strip().lower()
            # Remove trailing periods if any
            content = content.replace('.', '')
            our_results.append(content)
        else:
            print(f"Error {response.status_code}: {response.text}")
            our_results.append('')  # fallback

        # Logging
        print(f"Index  {int(count/3):03d}: ours='{our_results[-1]}', gt='{gt_results[count]}'")
        print(f"Normalized: ours='{normalize_output(our_results[-1])}', "
              f"gt='{normalize_output(gt_results[count])}'")
        if (count+1) % 3 == 0:
            print('---')

        # Build an entry to record in JSON
        record = {
            "index": count,
            "model": request_payload["model"],
            "question": question_list[count],
            "our_answer_raw": our_results[-1],
            "our_answer_normalized": normalize_output(our_results[-1]),
            "gt_answer_raw": gt_results[count],
            "gt_answer_normalized": normalize_output(gt_results[count])
        }
        inference_records.append(record)

        count += 1

# ================== EVALUATION & RESULTS ================== #
matches, total_num, matches_details = compare_outputs(our_results, gt_results)
match_array = np.array(matches_details).reshape([-1, 3])
col_sums = np.mean(match_array, axis=0)
score_name = "ViLP-F" if not args.without_fact else "ViLP-P"
print('\n')
print('---')
print(f"Model: {args.openai_model}")
print(f"{score_name} Score: {np.mean(col_sums[1:3]):.2f}")
print(f"{score_name} Prior: {col_sums[0]:.2f}")
row_sums = np.sum(match_array, axis=1)

per_image_stats = {}
per_image_stats["position_accuracy"] = np.mean(match_array, axis=0).tolist()
per_image_stats["count_of_correct_by_question"] = {
    "0_correct": int(np.sum(row_sums == 0)),
    "1_correct": int(np.sum(row_sums == 1)),
    "2_correct": int(np.sum(row_sums == 2)),
    "3_correct": int(np.sum(row_sums == 3))
}

final_json_output = {
    f"{score_name} Score": f"{np.mean(col_sums[1:3]):.2f}",
    f"{score_name} Prior": f"{col_sums[0]:.2f}",
    "per_image_stats": per_image_stats,
    "inference_records": inference_records
}

os.makedirs('ViLP_results', exist_ok=True)
output_filename = f"{score_name}_{args.openai_model}.json"
output_file_path = os.path.join('ViLP_results', output_filename)
with open(output_file_path, "w") as f:
    json.dump(final_json_output, f, indent=2)
print(f"Evaluation results have been saved to '{output_file_path}'.")
