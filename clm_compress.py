# %%
from datasets import load_dataset, Dataset
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig, Trainer, GenerationConfig, AutoModel
from tqdm.auto import tqdm
import torch
import numpy as np
import pandas as pd

# %%
model = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Llama-2-13b-chat-hf',
    load_in_4bit=True,
    device_map=0,
    trust_remote_code=True,
    cache_dir = '/nfs/nas-6.1/wclu/cache',
    torch_dtype=torch.bfloat16,
    max_memory = {i: '48000MB' for i in range(torch.cuda.device_count())},
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4'
    )
)
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-13b-chat-hf',  cache_dir='/nfs/nas-6.1/wclu/cache', padding_side='left')
tokenizer.pad_token = tokenizer.eos_token

# %%
def to_tokens_and_logprobs(model, tokenizer, input_texts):
    inputs = tokenizer(input_texts, padding='longest', return_tensors="pt").to('cuda:0')
    outputs = model(**inputs)
    probs = torch.log_softmax(outputs.logits, dim=-1).detach()

    probs = probs[:, :-1, :]
    input_ids = inputs.input_ids[:, 1:]
    gen_probs = torch.gather(probs, 2, input_ids[:, :, None]).squeeze(-1)

    batch = []
    for input_sentence, input_probs in zip(input_ids, gen_probs):
        text_sequence = []
        for token, p in zip(input_sentence, input_probs):
            if token not in tokenizer.all_special_ids:
                text_sequence.append((int(token), tokenizer.decode(token), p.item()))
        batch.append(text_sequence)
    return batch

# %%

test_data = pd.read_json('test_data.json')
prompt_list = test_data['instruction'].tolist()

# %%
def parallel_compress(prompt, num_token):
    seq = to_tokens_and_logprobs(model, tokenizer, prompt)[0]
    token_prob = pd.DataFrame(seq, columns=['token', 'text', 'prob']).sort_values('prob').reset_index()
    compressed_prompt = tokenizer.decode(token_prob.iloc[:num_token].sort_values('index')['token'].tolist())
    return compressed_prompt

# %%
def sequential_compress(prompt, time):
    if time == 0:
        return prompt
    else:
        seq = to_tokens_and_logprobs(model, tokenizer, prompt)[0]
        token_prob = pd.DataFrame(seq, columns=['token', 'text', 'prob'])
        compressed_prompt = tokenizer.decode(token_prob.drop(token_prob['prob'].idxmax())['token'].tolist())
        return sequential_compress(compressed_prompt, time-1)


batch_size = 16
full_result = []
for batch in tqdm([prompt_list[i:i+batch_size] for i in range(0, len(prompt_list), batch_size)]):
    inputs = tokenizer(batch, return_tensors="pt", padding='longest').to('cuda')
    outputs = model.generate(
        **inputs, 
        generation_config=GenerationConfig(
            do_sample=False,
            max_new_tokens = 300,
        )
    )
    texts = [t for t in tokenizer.batch_decode(outputs, skip_special_tokens=True)]
    full_result += texts
for i in range(len(prompt_list)):
    full_result[i] = full_result[i].replace(prompt_list[i],'')
# %%
def compress(compress_percentage):
    ratio = 1/(1-compress_percentage)
    seq_com = []
    for prompt in tqdm(prompt_list):
        token_len = len(tokenizer(prompt)['input_ids'])-1
        num_token = int(token_len//ratio)
        seq_com.append(sequential_compress(prompt, token_len-num_token))

    para_com = []
    for prompt in tqdm(prompt_list):
        token_len = len(tokenizer(prompt)['input_ids'])-1
        num_token = int(token_len//ratio)
        para_com.append(parallel_compress(prompt, num_token))

    seq_result = []
    for batch in tqdm([seq_com[i:i+batch_size] for i in range(0, len(prompt_list), batch_size)]):
        inputs = tokenizer(batch, return_tensors="pt", padding='longest').to('cuda')
        outputs = model.generate(
            **inputs, 
            generation_config=GenerationConfig(
                do_sample=False,
                max_new_tokens = 300,
            )
        )
        texts = [t for t in tokenizer.batch_decode(outputs, skip_special_tokens=True)]
        seq_result += texts

    para_result = []
    for batch in tqdm([para_com[i:i+batch_size] for i in range(0, len(prompt_list), batch_size)]):
        inputs = tokenizer(batch, return_tensors="pt", padding='longest').to('cuda')
        outputs = model.generate(
            **inputs, 
            generation_config=GenerationConfig(
                do_sample=False,
                max_new_tokens = 300,
            )
        )
        texts = [t for t in tokenizer.batch_decode(outputs, skip_special_tokens=True)]
        para_result += texts

    for i in range(len(seq_com)):
        seq_result[i] = seq_result[i].replace(seq_com[i],'')
    for i in range(len(para_com)):
        para_result[i] = para_result[i].replace(para_com[i],'')

    result_df = pd.DataFrame({
        'prompt': prompt_list,
        'seq_com_prompt': seq_com,
        'para_com_prompt': para_com,
        'output': full_result,
        'seq_com_output': seq_result,
        'para_com_output': para_result 
    })
    result_df.to_json(f'13b_{compress_percentage}.json')


for i in tqdm(range(1,10)):
    compress(i*0.1)