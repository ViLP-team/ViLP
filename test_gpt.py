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
    # Open the image directly from the raw bytes
    with Image.open(BytesIO(image_bytes)) as img:
        img = img.convert("RGB")
        img = img.resize(size, Image.LANCZOS)

        # Re-encode as JPEG in memory
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        buffer.seek(0)
        encoded_jpeg = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return encoded_jpeg

# ================== REQUEST PREPARATION ================== #
def prepare_requests(df):
    """
    Convert each row of the dataframe into multiple requests 
    (3 images -> 3 requests). Returns:
      - requests_batch: a list of request payloads
      - gt_results: the ground-truth answers for each image
    """
    requests_batch = []
    gt_results = []

    for i in range(len(df)):
        row = df.iloc[i]
        question = row['question']
        # Each row has up to 3 images & answers
        image_cols = ['image1', 'image2', 'image3']
        answer_cols = ['answer1', 'answer2', 'answer3']

        for img_col, ans_col in zip(image_cols, answer_cols):
            image_bytes = row[img_col]  # now already bytes
            gt_answer = row[ans_col]
            if pd.isna(image_bytes) or pd.isna(gt_answer):
                # skip if missing data
                continue

            processed_img_base64 = process_image(image_bytes)

            # Append ground truth to the list for evaluation
            gt_results.append(gt_answer)

            # Build request for OpenAI API
            request_payload = {
                "model": "gpt-4o-mini-2024-07-18",  # or whichever GPT model you need
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

    return requests_batch, gt_results

requests_batch, gt_results = prepare_requests(df)

# ================== API REQUEST EXECUTION ================== #
batch_size = len(df)  # or tweak as needed
our_results = []
count = 0

print('\nStart to inference:')
print('---')
for i in range(0, len(requests_batch), batch_size):
    batch = requests_batch[i : i + batch_size]
    for request_payload in batch:
        response = requests.post(API_URL, headers=headers, json=request_payload)
        if response.status_code == 200:
            result = response.json()
            # Extract the model's output (assuming 'content' is a string with the answer)
            content = result['choices'][0]['message']['content'].strip().lower()
            # Remove trailing periods if any
            content = content.replace('.', '')
            our_results.append(content)
        else:
            print(f"Error {response.status_code}: {response.text}")
            our_results.append('')  # fallback

        print(f"{count:03d}: ours='{our_results[-1]}', gt='{gt_results[count]}'")
        print(f"Normalized : ours='{normalize_output(our_results[-1])}', gt='{normalize_output(gt_results[count])}'")
        print('---')
        count += 1

# ================== EVALUATION & RESULTS ================== #
matches, total_num, matches_details = compare_outputs(our_results, gt_results)
print(f"\nMatching outputs: {matches}/{total_num}")
print(f"Percentage match: {matches / total_num * 100:.2f}%")

# Compute per-image accuracy in sets of 3
if total_num % 3 == 0:
    match_array = np.array(matches_details).reshape([-1, 3])
    print(f"Per-image position accuracy: {np.mean(match_array, axis=0)}")
    # Count how many rows had 0,1,2,3 correct
    row_sums = np.sum(match_array, axis=1)
    print(
        "Count of 0/1/2/3 correct images per question:",
        np.sum(row_sums == 0),
        np.sum(row_sums == 1),
        np.sum(row_sums == 2),
        np.sum(row_sums == 3)
    )