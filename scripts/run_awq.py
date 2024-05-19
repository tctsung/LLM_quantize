import logging
import pandas as pd
# load keys:
from dotenv import dotenv_values
from huggingface_hub import login
ENV_VAR = dotenv_values("../env/.env")
HF_key = ENV_VAR['HF_key']
login(token=HF_key) 
from huggingface_hub import HfApi
from huggingface_hub import create_repo
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM  # HF models
from awq import AutoAWQForCausalLM
# load self-written funcs:
import sys
sys.path.append("func/")      # add path 
import prompts
def main():   # run by `python scripts/run_awq.py` when located at the repo
	
    # load tiny llama model from HF API & save model to local for future comparison
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tinyllama_path = "../models/TinyLlama-1.1B-chat-v1.0"
    tinyllama, tokenizer = get_hf_model(model_id, tinyllama_path)

    # load models to autoawq module:
    
    # prepare quantization args:
    calib_data = prep_data('data/restaurant_chat_2024-05-11_21_38_18_603307.csv', tokenizer)
    quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }
    logging.INFO(calib_data[5])
    output_dir = '../models/TinyLlama-1.1B-chat-v1.0-awq'
    
    # do quantization:
    awq_quantization(tinyllama_path, tokenizer, quant_config, calib_data, output_dir)
    
    # push quantized model up to HF:
    create_repo("tctsung/TinyLlama-1.1B-chat-v1.0-awq", repo_type="model")
    api = HfApi()
    api.upload_folder(
        folder_path='../models/TinyLlama-1.1B-chat-v1.0-awq',
        repo_id="tctsung/TinyLlama-1.1B-chat-v1.0-awq",
        repo_type="model"
    )
def awq_quantization(model_path, tokenizer, quant_config, calib_data, output_dir):
    model = AutoAWQForCausalLM.from_pretrained(model_path)
    model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_data)
    model.save_quantized(output_dir)
    tokenizer.save_pretrained(output_dir)

def get_hf_model(model_id, save_folder):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, padding_side='left')
    model.save_pretrained(save_folder)
    tokenizer.save_pretrained(save_folder)
    return model, tokenizer

def prep_data(file_path, tokenizer):
    calib_df = pd.read_csv(file_path)
    calib_data = []
    for index, row in calib_df.iterrows():
        if index % 2 == 0:
            sys_msg = prompts.SYSTEM_MESSAGES['restaurant']
        else:
            sys_msg = prompts.SYSTEM_MESSAGES['default']
        msg = [
            {"role": "system", "content": sys_msg,},
            {"role": "user",  "content": row['user_input']},
            {'role': 'assistant', 'content': row['model_output']}
        ]
        input_text = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
        calib_data.append(input_text.strip())
    return calib_data
if __name__ == '__main__':
	main()      # the func to run