# ViLP
This repo is the codebase for the paper [Probing Visual Language Priors in VLMs](https://arxiv.org/abs/2501.00569). We explore the visual language priors in VLMs by constructing Question-Image-Answer triplets that deliberately deviate from the training data distribution. Also, we proposed Image-DPO to encourage the model to use more visual inputs.

Please access ViLP dataset on [Huggingface](https://huggingface.co/datasets/ViLP/ViLP). 

![Dataset Sample](sample.png)

## Usage

Our benchmark evaluation does not require the involvement of other LLMs/VLMs due to the design of the single-word output. 

We provide the evaluation code for both the LLaVA-v1.5 ([test_llava.py](https://github.com/ViLP-team/ViLP/blob/main/test_llava.py)) and OpenAI models ([test_gpt.py](https://github.com/ViLP-team/ViLP/blob/main/test_gpt.py)). It can be also easily integrated into other VLM inference pipelines. If you encounter any issues while evaluating ViLP, please email our team or leave a issue!

- Please download the `test_llava.py` and `eval_utils.py`, and run with: `python test_gpt.py`.

- Please download the `test_gpt.py` and `eval_utils.py`, and run with: `python test_gpt.py --api_key 'YOUR-OPENAI-KEY'`.

- As noted in our paper, we evaluate performance in two modes: with the fact included in the question (ViLP-F) and without it (ViLP-P). You can specify the mode by adding the `--without_fact` flag. For example: `python test_gpt.py --without_fact`. 

The output will be similar to the example below, where Normalized refers to our script that standardizes the single-word output for comparison. The script prints the final scores, e.g., `ViLP-F Score: 0.35\n ViLP-F Prior: 0.76`, and saves the output to a JSON file, e.g., `Evaluation results have been saved to 'ViLP_results/ViLP-F_LLaVA-v1.5-7b-ImageDPO.json'`.


```
Columns in dataset: Index(['question', 'image1', 'answer1', 'image2', 'answer2', 'image3',
       'answer3'],
      dtype='object')
Total rows: 300

Start to inference:
---
000: ours='four', gt='4'
Normalized : ours='4', gt='4'
---
001: ours='six', gt='6'
Normalized : ours='6', gt='6'
---
002: ours='eight', gt='8'
Normalized : ours='8', gt='8'
---
003: ours='sphere', gt='Sphere'
Normalized : ours='round', gt='round'
---
004: ours='cuboid', gt='Cube'
Normalized : ours='cube', gt='cube'
```

## Citation Information

If you find our data, code or paper useful, please consider citing:

```
@article{luo2024probing,
      title={Probing Visual Language Priors in VLMs},
      author={Luo, Tiange and Cao, Ang and Lee, Gunhee and Johnson, Justin and Lee, Honglak},
      journal={arXiv preprint arXiv:2501.00569},
      year={2024},
      url={https://arxiv.org/abs/2501.00569}
}
```
