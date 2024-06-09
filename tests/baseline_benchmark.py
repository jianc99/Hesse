import time
import torch
import sys
sys.path.append("..")
import torch.distributed as dist
from Hesse.Engine.utils import initialized_dist_baseline, make_causal_mask, setup_seed
from Hesse.Data.data_converter import convert_wiki_dataset, convert_cnn_dataset, convert_c4_dataset
from Hesse.Engine.llm_pipe import LLMEngine
from transformers import LlamaTokenizer, DataCollatorForLanguageModeling
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import argparse
from Hesse.Tree.utils import get_sampling_logits
from torch.nn.functional import softmax

parser = argparse.ArgumentParser(description='Process model configuration and partitions.')
parser.add_argument('--model', type=str, default="meta-llama/Llama-2-7b-hf", help='Model identifier.')
parser.add_argument('--B', type=int, default=1, help='Batch size.')
parser.add_argument('--M', type=int, default=256, help='Maximum length.')
parser.add_argument('--seed', type=int, default=123, help='Random seed.')
parser.add_argument('--dataset', type=str, default="cnn", help='dataset path')
parser.add_argument('--start', type=int, default=0, help='Dataset start index.')
parser.add_argument('--end', type=int, default=200, help='Dataset end index.')
parser.add_argument('--top_p', type=float, default=0.9, help='Target sample top_p.')
parser.add_argument('--temperature', type=float, default=0.6, help='Target sample temperature.')
args = parser.parse_args()

global_group=initialized_dist_baseline()
print("="*80)
global_rank=dist.get_rank()
if global_rank == 0:
    print(args)
setup_seed(args.seed)

MAX_LEN = args.M
MODEL_NAME = args.model
DTYPE = torch.float16
DEVICE = torch.device("cuda", global_rank)
BATCH_SIZE = args.B

engine = LLMEngine(max_length=MAX_LEN, model_name=MODEL_NAME, device=DEVICE, global_group=global_group, dtype=DTYPE, batch_size= BATCH_SIZE)
engine.initialize_cuda_graph([1])

tokenizer = LlamaTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
if args.dataset == 'wiki':
    tokenized_dataset_eval = convert_wiki_dataset(tokenizer=tokenizer).select(list(range(args.start, args.end)))
elif args.dataset == 'cnn':
    tokenized_dataset_eval = convert_cnn_dataset(tokenizer=tokenizer).select(list(range(args.start, args.end)))
else:
    tokenized_dataset_eval = convert_c4_dataset(tokenizer=tokenizer,file_path="../dataset/c4_small.json").select(list(range(args.start, args.end)))

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
dataloader = DataLoader(tokenized_dataset_eval, batch_size=BATCH_SIZE, collate_fn=data_collator, shuffle=False, drop_last=True)
num_eval_steps = len(dataloader)
total_time = 0.0
model_steps = 0
for step, batch in tqdm(enumerate(dataloader), total=num_eval_steps):
    input_ids = batch['input_ids'][..., :128].to(DEVICE)
    labels = batch['labels'][..., :128]
    terminate = False
    if (labels[:, -1] == -100)._is_any_true(): terminate = True
    output = input_ids.clone()
    prefix_len = input_ids.size(1)
    attention_mask = make_causal_mask((MAX_LEN, MAX_LEN), dtype=DTYPE, device=DEVICE)
    attention_mask = attention_mask[None, None, :, :].repeat(BATCH_SIZE,1,1,1)
    position_ids = torch.arange(prefix_len, device=DEVICE).unsqueeze(0)
    prefix_storage_ids = torch.arange(prefix_len, device=DEVICE)
    logits = engine.forward(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask[..., :prefix_len,:], storage_ids=prefix_storage_ids)
    seq_offset=prefix_len
    logits = get_sampling_logits(logits=logits[:,-1].clone(), top_p=args.top_p, T=args.temperature, replicate=False)
    logits = softmax(logits / args.temperature, dim=-1)
    next_tokens = logits.view(-1, 32000).multinomial(num_samples=1).view(BATCH_SIZE, 1)
    output = torch.cat((output, next_tokens),dim=-1)
    torch.cuda.synchronize()
    t1 = time.time()
    while output.size(1)<args.M and terminate == False:
        input_ids=next_tokens.clone()
        position_ids = torch.full((BATCH_SIZE,1),seq_offset, device=DEVICE)
        storage_ids = torch.tensor(seq_offset, device=DEVICE)
        logits = engine.forward(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask[..., seq_offset,:].unsqueeze(-2), storage_ids=storage_ids)
        logits = get_sampling_logits(logits=logits[:,-1].clone(), top_p=args.top_p, T=args.temperature, replicate=False)
        logits = softmax(logits / args.temperature, dim=-1)
        next_tokens = logits.view(-1, 32000).multinomial(num_samples=1).view(BATCH_SIZE, 1)
        output = torch.cat((output, next_tokens),dim=-1)
        seq_offset+=1
        model_steps+=1
        if (next_tokens[:,-1] == 2)._is_any_true() or (next_tokens[:,-1] == 0)._is_any_true(): terminate = True
    torch.cuda.synchronize()
    t2=time.time()
    total_time += t2-t1
    engine.llm.kv_cache.clear()
    if global_rank == 0:
        print(total_time/model_steps)
        for i in range(BATCH_SIZE):
            print(tokenizer.decode(output[i]))