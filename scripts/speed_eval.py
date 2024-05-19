from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset
import time
import pandas as pd
import sys   # for sys arg
import os
from dotenv import dotenv_values
from huggingface_hub import login

def main():   # run at colab
    # login:
    ENV_VAR = dotenv_values(".env")
    HF_key = ENV_VAR['HF_key']
    login(token=HF_key) 
    # load models & tokenizer:
    model_id = sys.argv[1]
    if "awq" in model_id:
         model = LLM(model = model_id, dtype='half', 
                    quantization='awq', gpu_memory_utilization=0.9, max_model_len=1024)
    else:
         model = LLM(model = model_id, dtype='float16', gpu_memory_utilization=0.9, max_model_len=1024)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # get input data:
    questions = load_dataset('lmsys/chatbot_arena_conversations')
    # randomly sampled 1000 questions:
    questions = questions['train']
    questions = questions.shuffle(seed=8)
    questions_random_n = questions['conversation_a'][:500]
    prompts = list(map(lambda x:x[0]['content'], questions_random_n))
    
    # speed test:
    speed_data = {'idx':[], 'token_len':[], 'time_spent':[],'token_per_sec':[]}
    sampling_params = SamplingParams(temperature=1.0,
                                 max_tokens=1024,
                                 min_p=0.5,
                                 top_p=0.85)
    for i, input_text in enumerate(prompts):
        if i % 10 == 0:
            print(f"Current progress: {i}/{len(prompts)}")
        token_len, time_spent = token_per_sec(model, tokenizer, input_text, sampling_params)
        speed_data['idx'].append(i)
        speed_data['token_len'].append(token_len)
        speed_data['time_spent'].append(time_spent)
        speed_data['token_per_sec'].append(token_len/time_spent)
    output = pd.DataFrame(speed_data)
    # save to mounted Google drive
    save_path = "/content/drive/MyDrive/Colab Notebooks/data/" + os.path.basename(model_id) + "_speed.csv"
    output.to_csv(save_path, index=False, encoding='utf-8-sig')
def token_per_sec(model, tokenizer, msg, sampling_params):
    chat_msg = [
            {"role": "system", "content": "You're a helpful assistant. Please provide a detailed and thorough answer to the following question."},
            {"role": "user",  "content": msg}
        ]
    prompt = tokenizer.apply_chat_template(chat_msg, tokenize=False, add_generation_prompt=False)
    start_time = time.time()
    output = model.generate(prompt, sampling_params)
    token_len = len(output[0].outputs[0].token_ids)
    time_spent = time.time() - start_time
    return token_len, time_spent
if __name__ == '__main__':
	main()      # the func to run