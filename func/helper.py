import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import logging
import sys
sys.path.append("D:/code/restaurant_LLM/helper/")
import prompts

def generate(model, tokenizer, input_text, generation_config, device, sys_msg_type='default'):
    model.to(device).eval()   # turn to inference mode
    input_text = prompts.llama_chat_prompt(input_text, sys_msg_type)
    stime = time.time()
    input_id = tokenizer(input_text)['input_ids']
    input_tensor = torch.tensor(
            input_id,
            device=device
        ).unsqueeze(0)
    output_ids = model.generate(
        input_tensor,
        generation_config = generation_config,
        pad_token_id=tokenizer.pad_token_id
    )
    generation_time = time.time() - stime
    output_text = tokenizer.decode(
        output_ids[0].tolist(),
        skip_special_tokens=True
    )
    # logging.INFO(f"Time spent: {str(generation_time)}")
    return output_text, generation_time
