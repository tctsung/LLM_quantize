# LLM_quantize
LLM quantization to speed up inference time of the restaurant recommendation chatbot project

Quantized model link: [Huggingface tctsung/TinyLlama-1.1B-chat-v1.0-awq](https://huggingface.co/tctsung/TinyLlama-1.1B-chat-v1.0-awq)

## Introduction

This repo focus on AWQ quantization of TinyLlama-1.1B-Chat-v1.0 model, and evaluation of model performance & inference speed-up after quantization.

Overall, after AWQ quantization, 

1. The TinyLlama-AWQ model inference speed improve to 1.62X with 140.47 new tokens generated per sec.

![Alt text](evaluation/Inference_speed.png)

2. The model size is compressed from 4.4GB to 0.78GB, only 17.57% memory footprint comparing to the original 

![Alt text](evaluation/Size.png)

3. Model performance (raw accuracy) is very close to original model using 6 types of LLM tasks. At most 1% accuracy degradation is observed.

![Alt text](evaluation/Accuracy.png)
![Alt text](evaluation/Accuracy_degradation.png)


## Inference tutorial

* You can check this 