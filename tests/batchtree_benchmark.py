import sys
sys.path.append("..")
import argparse
import time
import torch
import torch.distributed as dist
from transformers import DataCollatorForLanguageModeling, AutoTokenizer
from Hesse.Tree.BatchTree import BatchSTree
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from Hesse.Engine.llm_pipe import LLMEngine
from Hesse.Data.data_converter import convert_wiki_dataset, convert_cnn_dataset, convert_c4_dataset
from Hesse.Tree.utils import cuda_graph_for_sampling_argmax
from Hesse.Engine.utils import setup_seed, initialized_dist_spec

parser = argparse.ArgumentParser()
parser.add_argument('--model', default="JackFram/llama-68m", type=str, help='model')
parser.add_argument('--target', default="princeton-nlp/Sheared-LLaMA-1.3B", type=str, help='target model')
parser.add_argument('--growmap', type=str, default="demo_tree.pt", help='growmap path')
parser.add_argument('--dataset', type=str, default="cnn", help='dataset path')
parser.add_argument('--start', type=int, default=0, help='start')
parser.add_argument('--end', type=int, default=200, help='end')
parser.add_argument('--T', type=float, default=0.6, help='temperature')
parser.add_argument('--P', type=float, default=0.9, help='top_p')
parser.add_argument('--B', type=int, default=16, help='batch_size')
parser.add_argument('--M', type=int, default=256, help='max length')
parser.add_argument('--seed', type=int, default=123, help='random seed')
parser.add_argument('--Mode', type=str, default="fast")

# Target model information
parser.add_argument('--target_group', nargs='+', type=int, help='Target group of ranks')
# Draft model information
parser.add_argument('--draft_group', nargs='+', type=int, help='Target group of ranks')
args = parser.parse_args()
# print(args)
setup_seed(args.seed)
target_global_group, draft_global_group = initialized_dist_spec(args)
global_rank=dist.get_rank()
BATCH_SIZE = args.B

def simulation_fast(target_model : LLMEngine, draft_model: LLMEngine, dataloader: DataLoader, T=0.6, top_p=0.9, 
            max_length=512, grow_map=None, sampling_callables = None,
            sample_gather_indices = None):
    num_eval_steps = len(dataloader)
    num_decoding_steps = 0
    num_large_model_steps = 0
    total_time = 0.0
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=False)
    with torch.no_grad():
        for step, batch in tqdm(enumerate(dataloader), total=num_eval_steps):
            input_ids = batch['input_ids'][..., :128]
            labels = batch['labels'][..., :128]
            terminate = False
            if (labels[:, -1] == -100)._is_any_true(): terminate = True
            spectree = BatchSTree (prefix=input_ids, device=DEVICE, temperature=T, top_p=top_p,
                                    draft_model_engine=draft_model, target_model_engine=target_model, max_length=max_length, grow_map=grow_map,
                                   sampling_callables=sampling_callables, sample_gather_indices = sample_gather_indices, batch_size=BATCH_SIZE, max_target_seq=256
                                    )
            torch.cuda.synchronize()
            t1 = time.time()
            longest=128

            while longest < 256 and terminate == False:
                spectree.construct_grow_map()
                num_nodes, terminate = spectree.verify()
                longest = num_nodes.max()
                num_large_model_steps += 1

            torch.cuda.synchronize()
            t2 = time.time()
            num_decoding_steps += (num_nodes.sum() - input_ids.shape[1]*BATCH_SIZE)
            total_time += (t2 - t1)
            if dist.get_rank() == 0:
                for i in range(BATCH_SIZE):
                    print(tokenizer.decode(spectree.tokens[i,:num_nodes[i]]))
                print("total time :{:.5f}s, latency :{:.5f}s, decoding step: {}, large model step: {}".format(total_time, total_time / num_decoding_steps, num_decoding_steps, num_large_model_steps))
            draft_model.llm.kv_cache.clear()
            target_model.llm.kv_cache.clear()
    return num_decoding_steps / num_large_model_steps

def simulation_benchmark(target_model : LLMEngine, draft_model: LLMEngine, dataloader: DataLoader, T=0.6, top_p=0.9, 
            max_length=512, grow_map=None, sampling_callables = None,
            sample_gather_indices = None):
    num_eval_steps = len(dataloader)
    num_decoding_steps = 0
    num_large_model_steps = 0
    initialize_time = 0.0
    speculate_time = 0.0
    verify_time = 0.0
    large_model_run = 0.0
    accept_loop = 0.0
    kv_select = 0.0
    sample_time = 0.0
    small_model_compute = 0.0
    with torch.no_grad():
        for step, batch in tqdm(enumerate(dataloader), total=num_eval_steps):
            input_ids = batch['input_ids'][..., :128]
            labels = batch['labels'][..., :128]
            terminate = False
            if (labels[:, -1] == -100)._is_any_true(): terminate = True
            spectree = BatchSTree (prefix=input_ids, device=DEVICE, temperature=T, top_p=top_p,
                                    draft_model_engine=draft_model, target_model_engine=target_model, max_length=max_length, grow_map=grow_map,
                                   sampling_callables=sampling_callables, sample_gather_indices = sample_gather_indices, batch_size=BATCH_SIZE, max_target_seq=256
                                    )
            longest=128

            while longest < 256 and terminate == False:
                torch.cuda.synchronize()
                t1 = time.time()
                torch.cuda.synchronize()
                t2 = time.time()
                a, b = spectree.construct_grow_map(benchmark=True)
                torch.cuda.synchronize()
                t3 = time.time()
                num_nodes,x, y, z, terminate = spectree.verify(benchmark=True)
                torch.cuda.synchronize()
                t4 = time.time()
                longest = num_nodes.max()
                sample_time += a
                small_model_compute += b
                large_model_run += x
                accept_loop += y
                kv_select += z
                initialize_time += (t2 - t1)
                speculate_time += (t3 - t2)
                verify_time += (t4 - t3)
                num_large_model_steps += 1
            num_decoding_steps += (num_nodes.sum() - input_ids.shape[1]*BATCH_SIZE)
            draft_model.llm.kv_cache.clear()
            target_model.llm.kv_cache.clear()
            if num_large_model_steps > 0 and global_rank==0:
                print(num_decoding_steps / num_large_model_steps)
            if global_rank == 0:
                print("total decoding steps: {}".format(num_decoding_steps), "large model steps: {}".format(num_large_model_steps), "avg decoding step: {}".format(num_decoding_steps / num_large_model_steps))
                print("initialization time:{}".format(initialize_time / num_large_model_steps), "speculate time: {}".format(speculate_time / num_large_model_steps),  "verify time: {}".format(verify_time / num_large_model_steps))
                print("large model run: {}".format(large_model_run / num_large_model_steps) , "accept loop: {}".format(accept_loop / num_large_model_steps), "kv select: {}".format(kv_select / num_large_model_steps))
                print("small model run: {}".format(small_model_compute / num_large_model_steps) , "sample time: {}".format(sample_time / num_large_model_steps))
    return num_decoding_steps / num_large_model_steps

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

if args.dataset == 'wiki':
    tokenized_dataset_eval = convert_wiki_dataset(tokenizer=tokenizer).select(list(range(args.start, args.end)))
elif args.dataset == 'cnn':
    tokenized_dataset_eval = convert_cnn_dataset(tokenizer=tokenizer).select(list(range(args.start, args.end)))
else:
    tokenized_dataset_eval = convert_c4_dataset(tokenizer=tokenizer,file_path="../dataset/c4_small.json").select(list(range(args.start, args.end)))

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
dataloader = DataLoader(tokenized_dataset_eval, batch_size=BATCH_SIZE, collate_fn=data_collator, shuffle=False, drop_last=True)

path = args.growmap
grow_map = torch.load(path)
tree_size = grow_map["size"]
idx_lists = grow_map["roots"]
branch_lists = grow_map['branches']
draft_step = len(grow_map["roots"])

MAX_LEN = args.M + tree_size
TARGET_MODEL_NAME = args.target
DRAFT_MODEL_NAME = args.model
DTYPE = torch.float16
DEVICE = torch.device("cuda", global_rank)

sampling_callables = {}
sample_gather_indices = {}
for i in range(draft_step - 1):
    idx_len = len(idx_lists[i])
    num_samples = max(branch_lists[i])
    sampling_callables[i] = cuda_graph_for_sampling_argmax(device=DEVICE,
        max_length=args.M, idx_len=idx_len, num_samples=num_samples,
        temperature=args.T, tree_size=tree_size)  
for i in range(draft_step - 1):
    ith_gather_list = []
    max_num_samples = max(branch_lists[i])
    for j, branch in enumerate(branch_lists[i]):
        branch_index = torch.arange(branch, device=DEVICE, dtype=torch.long)
        branch_index = branch_index + j * max_num_samples
        ith_gather_list.append(branch_index)
    ith_gather_list = torch.cat(ith_gather_list)
    sample_gather_indices[i] = ith_gather_list

cg_list_target = [tree_size]
cg_list_draft = [sum(x) for x in branch_lists]
cg_list_draft.append(1)

draft_model =  LLMEngine(max_length=MAX_LEN, model_name=DRAFT_MODEL_NAME, device=DEVICE, batch_size=BATCH_SIZE, dtype=torch.float16, global_group=draft_global_group)
draft_model.initialize_cuda_graph(cg_list_draft)
target_model = LLMEngine(max_length=MAX_LEN, model_name=TARGET_MODEL_NAME, device=DEVICE, batch_size=BATCH_SIZE, dtype=torch.float16, global_group=target_global_group)
target_model.initialize_cuda_graph(cg_list_target)

if args.Mode == "fast":
    simulation_fast(target_model=target_model, draft_model=draft_model, dataloader=dataloader, T=args.T, top_p=args.P,
                                     max_length=MAX_LEN, grow_map = grow_map, sampling_callables=sampling_callables, sample_gather_indices = sample_gather_indices)
else:
    simulation_benchmark(target_model=target_model, draft_model=draft_model, dataloader=dataloader, T=args.T, top_p=args.P,
                                     max_length=MAX_LEN, grow_map = grow_map, sampling_callables=sampling_callables, sample_gather_indices = sample_gather_indices)

