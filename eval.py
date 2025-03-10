import argparse
import os
import logging
import torch
import pandas as pd
import transformers




def read_data(path):
    data = pd.read_parquet(path)
    return data

def write_data(data, kv, path):
    for key, value in kv.items():
        data[key] = value
    data.to_parquet(path)

def data2message(data):
    context, question, choices_ = data['context'], data['question'], data['choices']
    choices = ''
    labels = ['A', 'B', 'C', 'D']
    for i in range(len(choices_)):
        choices += labels[i] + '. '+ choices_[i] + '\n'   
    message = 'Context: \n' + context + '\n' + 'Question: ' + question + '\n' + choices
    return message

def instruction():
    instruction = "You are an expert in reading comprehension. \
Read the passage below and select one of the most appropriate options to answer the question. \
You MUST give one option, and just give the option directly, without any explanation."
    return instruction


def call_pipeline(pipeline, instruction, message):
    input = {
        {'role': 'system', 'content': instruction},
        {'role': 'user', 'content': message}
    }
    prompt = pipeline.tokenizer.apply_chat_template(
            input, 
            tokenize=False, 
            add_generation_prompt=True
    )
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    outputs = pipeline(
        prompt,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=False,
        pad_token_id = pipeline.tokenizer.eos_token_id,
    )    
    response = outputs[0]["generated_text"][len(prompt):]
    return response

def parser(response, answer):
    label = response[0]
    return label, label == answer

def eval(data, model_id):
    logging.info(f'[START MODEL]: {model_id}')
    instruction = instruction()
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device='auto'
    ) 
    res = []
    labels = []
    correct, total = 0, len(data)
    for i in range(len(data)):
        instance = data.iloc[i]
        message = data2message(instance)
        response = call_pipeline(pipeline, instruction, message)
        label, pred = parser(response, instance['answer'])
        res.append(response)
        labels.append(label)
        correct += pred
        logging.info(f'[index]: {i}/{total}, {response}')
    logging.info(f'[ACC]: {correct / total}')
    return response, labels



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str)
    args = parser.parse_args()
    model_id = args.name

    data = read_data('data/MRCEval.parquet')
    res, labels = eval(data, model_id)
    res_path = 'res/response.parquet'
    results = {
        model_id: res,
        'labels': labels
    }

    if not os.path.exists('res'):
        os.makedirs('res')
    write_data(data, results, res_path)



if __name__ == "__main__":
    main()
