import argparse
import json
import math
import os
import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from PIL import Image
from io import BytesIO
from huggingface_hub import hf_hub_download

from llava.constants import (
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX
)
from llava.conversation import SeparatorStyle, conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    tokenizer_image_token,
)

# Your local evaluation utilities
from eval_utils import compare_outputs, normalize_output

def process_image_bytes(image_bytes, image_processor):
    """
    Convert raw image bytes into a preprocessed tensor for LLaVA.
    """
    with Image.open(BytesIO(image_bytes)) as img:
        img = img.convert("RGB")
        # Preprocess for LLaVA
        image_tensor = (
            image_processor.preprocess(img, return_tensors="pt")["pixel_values"]
            .half()
            .cuda()
        )
    return image_tensor


def run_inference(df, model, tokenizer, image_processor, conv_mode, args):
    """
    Loop over the ViLP dataframe rows and perform inference for 1..3 images per row.
    Return the list of predictions (our_results) and ground truths (gt_results).
    """
    our_results = []
    gt_results = []

    # Prepare conversation template
    conv_template = conv_templates[conv_mode].copy()
    inference_records = []

    print("\nStart inference on ViLP dataset:")
    for i in range(len(df)):
        row = df.iloc[i]
        question = row["question"]

        # we remove the fact and leave question solely for ViLP-P score.   
        if args.without_fact:
            question = question.split('.')[1].strip()
        print('---')

        # Collect each of the up to 3 images
        for img_col, ans_col in zip(["image1", "image2", "image3"],
                                    ["answer1", "answer2", "answer3"]):
            if pd.isna(row[img_col]) or pd.isna(row[ans_col]):
                # Skip if the image or the answer is missing
                continue

            # Process the image
            image_tensor = process_image_bytes(row[img_col], image_processor)

            # Construct the question prompt
            # e.g., "Please answer with one word: {question}"

            user_prompt = f"Please answer with one word: {question}"
            # Copy a fresh conversation template each time
            conv = conv_template.copy()

            # 1) Add [IMAGE] token
            # 2) Add the user question
            inp = DEFAULT_IMAGE_TOKEN + "\n" + user_prompt
            conv.append_message(conv.roles[0], inp)  # user
            conv.append_message(conv.roles[1], None) # assistant
            prompt = conv.get_prompt()

            # Tokenize prompt (with special image placeholder)
            input_ids = tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).cuda()

            # Prepare stopping criteria
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            # Generate
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=[image_tensor],
                    do_sample=(args.temperature > 0),
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=1024,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                )

            # Decode the output
            input_token_len = input_ids.shape[1]
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0].strip()

            # Store prediction and ground truth
            pred = outputs.lower().strip().replace('.', '')
            our_results.append(pred)
            gt_results.append(str(row[ans_col]))  # ground truth

            # Print intermediate results for debugging
            print(f"Index  {i:03d}: pred='{pred}', gt='{row[ans_col]}'")
            print(f"Normalized: pred='{normalize_output(pred)}', gt='{normalize_output(row[ans_col])}'")
                    # Build an entry to record in JSON
            record = {
                "index": i,
                "model": args.model_path,
                "question": question,
                "our_answer_raw": pred,
                "our_answer_normalized": normalize_output(pred),
                "gt_answer_raw": row[ans_col],
                "gt_answer_normalized": normalize_output(row[ans_col])
            }
            inference_records.append(record)


    return our_results, gt_results, inference_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="ViLP/LLaVA-v1.5-7b-ImageDPO",
                        help="Hugging Face path to the LLaVA model checkpoint.")
    parser.add_argument("--model-base", type=str, default=None,
                        help="If your model requires a base (e.g. LLaMA) path, specify here.")
    parser.add_argument("--conv-mode", type=str, default="llava_v1",
                        help="Which conversation template to use (llava_v1, etc.).")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Nucleus sampling top-p.")
    parser.add_argument("--without_fact", action='store_true', default=False,
                        help="Whether to include the fact sentence in the question."
    )
    args = parser.parse_args()

    # ================== HUGGING FACE LOAD (ViLP dataset) ================== #
    parquet_path = hf_hub_download(
        repo_id="ViLP/ViLP",
        filename="ViLP.parquet",
        repo_type="dataset",
        use_auth_token=True  # or False if not needed
    )
    df = pd.read_parquet(parquet_path)
    print("Columns in dataset:", df.columns)
    print("Total rows in dataset:", len(df))

    # ================== LOAD LLaVA MODEL ================== #
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=args.model_base,
        model_name=model_name
    )
    model.cuda().eval()

    # ================== RUN INFERENCE ================== #
    our_results, gt_results, inference_records = run_inference(df, model, tokenizer, image_processor, args.conv_mode, args)

    # ================== EVALUATION ================== #
    matches, total_num, matches_details = compare_outputs(our_results, gt_results)
    match_array = np.array(matches_details).reshape([-1, 3])
    col_sums = np.mean(match_array, axis=0)
    score_name = "ViLP-F" if not args.without_fact else "ViLP-P"
    print('\n')
    print('---')
    print(f"Model: {args.model_path}")
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
    output_filename = f"{score_name}_{args.model_path.split('/')[-1]}.json"
    output_file_path = os.path.join('ViLP_results', output_filename)
    with open(output_file_path, "w") as f:
        json.dump(final_json_output, f, indent=2)
    print(f"Evaluation results have been saved to '{output_file_path}'.")


if __name__ == "__main__":
    main()
