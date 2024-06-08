import sys
sys.path.append("..")
import copy
import argparse
import time
import torch
import torch.distributed as dist
from transformers import DataCollatorForLanguageModeling, AutoTokenizer
from Hesse.Tree.PipeTree import PipeTree_Draft, PipeTree_Target
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from Hesse.Engine_pipe.pipleline import LLM_Pipeline
from Hesse.Data.data_converter import convert_wiki_dataset, convert_cnn_dataset, convert_c4_dataset
from Hesse.Tree.utils import cuda_graph_for_sampling_argmax
from Hesse.Engine_pipe.utils import setup_seed, initialized_dist_pipe

parser = argparse.ArgumentParser()
parser.add_argument('--model', default="JackFram/llama-68m", type=str, help='model')
parser.add_argument('--target', default="princeton-nlp/Sheared-LLaMA-1.3B", type=str, help='target model')
parser.add_argument('--growmap', type=str, default="1.3b-70b_tree.pt", help='growmap path')
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
parser.add_argument('--target_layer_partition', nargs='+', type=int, help='Layer partitioning as a list of integers.')
parser.add_argument('--target_tp_groups', nargs='+', type=int, help='TP groups as a list of integers.')
parser.add_argument('--target_group', nargs='+', type=int, help='Target group of ranks')
# Draft model information
parser.add_argument('--draft_layer_partition', nargs='+', type=int, help='Layer partitioning as a list of integers.')
parser.add_argument('--draft_tp_groups', nargs='+', type=int, help='TP groups as a list of integers.')
parser.add_argument('--draft_group', nargs='+', type=int, help='Target group of ranks')

args = parser.parse_args()
# print(args)
# time.sleep(1000)
setup_seed(args.seed)
target_pp_config, draft_pp_config = initialized_dist_pipe(args)
# dist.barrier()
# print(target_pp_config,draft_pp_config)
global_rank=dist.get_rank()
BATCH_SIZE = args.B

def simulation_fast(draft_model: LLM_Pipeline, dataloader: DataLoader, max_length=512, grow_map=None, sampling_callables = None, sample_gather_indices = None, target_rank0=0, draft_rank0=0):
    num_eval_steps = len(dataloader)
    num_decoding_steps = 0
    num_large_model_steps = 0
    total_time = 0.0
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=False)
    batch_tree_1 = None
    batch_tree_2 = None
    with torch.no_grad():
        for step, batch in tqdm(enumerate(dataloader), total=num_eval_steps):
            input_ids = batch['input_ids'][..., :128]
            labels = batch['labels'][..., :128]
            terminate = False
            if (labels[:, -1] == -100)._is_any_true(): terminate = True
            mini_batch_1 = input_ids[:BATCH_SIZE//2]
            mini_batch_2 = input_ids[BATCH_SIZE//2:]
            batch_tree_1 = PipeTree_Draft(draft_model_engine=draft_model, prefix=mini_batch_1, max_length=max_length, device=DEVICE, batch_size=BATCH_SIZE//2, grow_map=grow_map, sampling_callables=sampling_callables, sample_gather_indices= sample_gather_indices, target_rank0=target_rank0, draft_rank0=draft_rank0, idx=0)
            batch_tree_2 = PipeTree_Draft(draft_model_engine=draft_model, prefix=mini_batch_2, max_length=max_length, device=DEVICE, batch_size=BATCH_SIZE//2, grow_map=grow_map, sampling_callables=sampling_callables, sample_gather_indices= sample_gather_indices, target_rank0=target_rank0, draft_rank0=draft_rank0, idx=1)
            num_nodes = torch.zeros(BATCH_SIZE,device=DEVICE).long()
            
            batch_tree_1.construct_grow_map()
            batch_tree_1.request_target()
            batch_tree_2.construct_grow_map()
            longest=128
            torch.cuda.synchronize()
            t1 = time.time()
            while longest < 256 and terminate == False:
                batch_tree_1.receive_result()
                num_large_model_steps+=1
                batch_tree_2.request_target()
                num_nodes[:BATCH_SIZE//2], terminate = batch_tree_1.verify()
                longest = num_nodes.max()
                if longest>= 256 or terminate == True:
                    batch_tree_2.receive_result()
                    num_large_model_steps+=1
                    num_nodes[BATCH_SIZE//2:], terminate = batch_tree_2.verify()
                    break
                batch_tree_1.construct_grow_map()
                batch_tree_2.receive_result()
                num_large_model_steps+=1
                batch_tree_1.request_target()
                num_nodes[BATCH_SIZE//2:], terminate = batch_tree_2.verify()
                longest = num_nodes.max()
                if longest>= 256 or terminate == True:
                    batch_tree_1.receive_result()
                    num_large_model_steps+=1
                    num_nodes[:BATCH_SIZE//2], terminate = batch_tree_1.verify()
                    break
                batch_tree_2.construct_grow_map()

            torch.cuda.synchronize()
            t2 = time.time()
            num_decoding_steps += (num_nodes.sum() - input_ids.shape[1]*BATCH_SIZE)
            total_time += (t2 - t1)
            if dist.get_rank() == draft_rank0:
                for i in range(BATCH_SIZE//2):
                    print(tokenizer.decode(batch_tree_1.tokens[i,:batch_tree_1.num_nodes[i]]))
                for i in range(BATCH_SIZE//2):
                    print(tokenizer.decode(batch_tree_2.tokens[i,:batch_tree_2.num_nodes[i]]))
                print("total time :{:.5f}s, latency :{:.5f}s, decoding step: {}, large model step: {}".format(total_time, total_time / num_decoding_steps, num_decoding_steps, num_large_model_steps))
            control_tensor = torch.tensor([4],device=DEVICE)
            dist.broadcast(control_tensor,draft_rank0)
            draft_model.clear_kv()
            batch_tree_1 = None
            batch_tree_2 = None
            dist.barrier()
            # time.sleep(100)
    return num_decoding_steps / num_large_model_steps

if target_pp_config!=None:
    # time.sleep(100)
    # dist.barrier()
    path = args.growmap
    grow_map = torch.load(path)
    tree_size = grow_map["size"]
    MAX_LEN = args.M + tree_size
    TARGET_MODEL_NAME = args.target
    DTYPE = torch.float16
    # DEVICE = torch.device("cuda", global_rank)
    DEVICE = torch.device("cuda", 0)
    cg_list_target = [tree_size]
    target_model = LLM_Pipeline(max_length=MAX_LEN, model_name=TARGET_MODEL_NAME, device=DEVICE, batch_size=BATCH_SIZE//2, dtype=torch.float16, pp_config=target_pp_config, cg_list=cg_list_target)
    dist.barrier()
    target_rank0 = args.target_group[0]
    draft_rank0 = args.draft_group[0]
    mini_batch_1_tree = None
    mini_batch_2_tree = None
    with torch.no_grad():
        while True:
            control_tensor = torch.tensor([0],device=DEVICE)
            dist.broadcast(control_tensor,draft_rank0)
            if control_tensor[0] == 0:
                prefix = torch.zeros((BATCH_SIZE//2,128), device=DEVICE).long()
                dist.broadcast(prefix, draft_rank0)
                mini_batch_1_tree = PipeTree_Target(device=DEVICE, target_model_engine=target_model,prefix=prefix, temperature=args.T, top_p=args.P,
                                        max_length=MAX_LEN, grow_map = grow_map, batch_size=BATCH_SIZE//2, target_rank0=target_rank0, draft_rank0=draft_rank0)
            elif control_tensor[0] == 1:
                prefix = torch.zeros((BATCH_SIZE//2,128), device=DEVICE).long()
                dist.broadcast(prefix, draft_rank0)
                mini_batch_2_tree = PipeTree_Target(device=DEVICE, target_model_engine=target_model,prefix=prefix, temperature=args.T, top_p=args.P,
                                        max_length=MAX_LEN, grow_map = grow_map, batch_size=BATCH_SIZE//2, target_rank0=target_rank0, draft_rank0=draft_rank0)
            elif control_tensor[0] == 2:
                mini_batch_1_tree.verify()
            elif control_tensor[0] == 3:
                mini_batch_2_tree.verify()
            elif control_tensor[0] == 4:
                target_model.clear_kv()
                mini_batch_1_tree = None
                mini_batch_2_tree = None
                dist.barrier()
elif draft_pp_config!=None:
    dist.barrier()
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
    DRAFT_MODEL_NAME = args.model
    DTYPE = torch.float16
    # DEVICE = torch.device("cuda", global_rank)
    DEVICE = torch.device("cuda", 0)

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

    cg_list_draft = [sum(x) for x in branch_lists]
    cg_list_draft.append(1)
    target_rank0 = args.target_group[0]
    draft_rank0 = args.draft_group[0]

    draft_model =  LLM_Pipeline(max_length=MAX_LEN, model_name=DRAFT_MODEL_NAME, device=DEVICE, batch_size=BATCH_SIZE//2, dtype=torch.float16, pp_config=draft_pp_config, cg_list=cg_list_draft)
    if args.Mode == "fast":
        simulation_fast(draft_model=draft_model, dataloader=dataloader, max_length=MAX_LEN, grow_map = grow_map, sampling_callables=sampling_callables, sample_gather_indices = sample_gather_indices, target_rank0 = target_rank0, draft_rank0 = draft_rank0)

