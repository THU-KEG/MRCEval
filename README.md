# MRCEval Benchmark
MRCEval is a comprehensive benchmark for machine reading comprehension (MRC) designed to assess the reading comprehension (RC) capabilities of LLMs, covering 13 sub-tasks with a total of 2.1K high-quality multi-choice questions.


## Quick Start
### Download dataset
MRCEval can be loaded from [Huggingface](https://huggingface.co/datasets/THU-KEG/MRCEval). Download and place the dataset file into the `data/`.

### Install dependencies
Create a python environment, then intall required dependencies:
```
pip install -r requirements.txt
```


### Choose a model
Choose a `model_id` from huggingface, such as `meta-llama/Llama-3.1-8B-Instruct`, or your own `model_path`.




### Evaluation
Run `eval.py`:
```
python eval.py --model [model_id]
```
