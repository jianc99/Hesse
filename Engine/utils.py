import torch
import torch.distributed as dist
import argparse
import numpy as np
import random

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    return mask

def args_parse_spec():
    parser = argparse.ArgumentParser(description='Process model configuration and partitions.')
    
    parser.add_argument('--target_model', type=str, default="meta-llama/Llama-2-70b-hf", help='Model identifier.')
    parser.add_argument('--draft_model', type=str, default="meta-llama/Llama-2-7b-hf", help='Model identifier.')
    parser.add_argument('--T', type=int, default=2000, help='Repeat times.')
    parser.add_argument('--B', type=int, default=1, help='Batch size.')
    parser.add_argument('--P', type=int, default=128, help='Prefix length.')
    parser.add_argument('--M', type=int, default=512, help='Maximum length.')
    parser.add_argument('--D', type=int, default=1, help='Decrement length.')
    parser.add_argument('--seed', type=int, default=123, help='Random seed.')
    parser.add_argument('--dataset', type=str, default="/dataset/c4_small.json", help='dataset path')
    parser.add_argument('--start', type=int, default=0, help='Dataset start index.')
    parser.add_argument('--end', type=int, default=200, help='Dataset end index.')
    # Sample parameters
    parser.add_argument('--top_k', type=int, default=20, help='Target samlple top_k')
    parser.add_argument('--top_p', type=float, default=0.9, help='Target sample top_p.')
    parser.add_argument('--temperature', type=float, default=0.6, help='Target sample temperature.')
    # Target model information
    parser.add_argument('--target_layer_partition', nargs='+', type=int, help='Layer partitioning as a list of integers.')
    parser.add_argument('--target_tp_groups', nargs='+', type=int, help='TP groups as a list of integers.')
    parser.add_argument('--target_group', nargs='+', type=int, help='Target group of ranks')
    # Draft model information
    parser.add_argument('--draft_layer_partition', nargs='+', type=int, help='Layer partitioning as a list of integers.')
    parser.add_argument('--draft_tp_groups', nargs='+', type=int, help='TP groups as a list of integers.')
    parser.add_argument('--draft_group', nargs='+', type=int, help='Target group of ranks')
    # Speculative decoding parameters
    parser.add_argument('--depth', type=int, default=1, help='Draft speculation length')
    args = parser.parse_args()
    
    return args

def gen_tp_rank_groups(tp_groups,rank_list):
    tp_rank_groups = []
    current_tp_rank = 0
    for tp_rank in tp_groups:
        new_tp_group = []
        for _ in range(tp_rank):
            new_tp_group.append(rank_list[current_tp_rank])
            current_tp_rank += 1
        tp_rank_groups.append(new_tp_group)
    return tp_rank_groups

def get_group_for_rank(rank, index_mapping):
    return index_mapping.get(rank)

def gen_include_layers(current_stage, layer_partition):
    start_idx = 0
    stage_indices = []
    for part in layer_partition:
        indices = list(range(start_idx, start_idx + part))
        stage_indices.append(indices)
        start_idx += part
    return stage_indices[current_stage]

def generate_index_mapping(original_lists):
    index_mapping = {}
    for index, group in enumerate(original_lists):
        for rank in group:
            index_mapping[rank] = index
    return index_mapping

def initialized_dist_spec(args):
    dist.init_process_group(backend='nccl')
    global_rank=dist.get_rank()
    # torch.cuda.set_device(0)
    torch.cuda.set_device(global_rank)

    target_group = args.target_group
    draft_group = args.draft_group

    target_commgroup=dist.new_group(target_group)
    draft_commgroup=dist.new_group(draft_group)
    return target_commgroup, draft_commgroup


def initialized_dist_baseline():
    dist.init_process_group(backend='nccl')
    global_rank=dist.get_rank()
    # torch.cuda.set_device(0)
    torch.cuda.set_device(global_rank)
    global_group = dist.new_group(list(range(dist.get_world_size())))
    return global_group

if __name__ == "__main__":
    # target_tp_groups=[4,4]
    # target_layer_partition=[40,40]
    # target_groups=[2, 3, 4, 5, 6, 7, 8, 9]
    # global_rank=3
    # target_stage_num = len(target_tp_groups)
    # target_tp_rank_groups = gen_tp_rank_groups(target_tp_groups,target_groups)
    # target_index_mapping = generate_index_mapping(target_tp_rank_groups)
    # target_current_stage = get_group_for_rank(global_rank,target_index_mapping)
    # target_current_stage_layers=gen_include_layers(target_current_stage,target_layer_partition)
    # target_pp_config={
    #     'num_stages':target_stage_num,
    #     'groups_indices':target_tp_rank_groups,
    #     'current_stage':target_current_stage,
    #     'current_layers':target_current_stage_layers,
    #     }
    # print(target_index_mapping)
    # print(target_pp_config)
    from transformers import LlamaTokenizer, DataCollatorForLanguageModeling
    from torch.utils.data.dataloader import DataLoader
    from accelerate import Accelerator
    from tqdm import tqdm
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_dataset_eval = convert_dataset(tokenizer=tokenizer,file_path="../dataset/c4_small.json").select(list(range(0,20)))
    print(tokenized_dataset_eval)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    dataloader = DataLoader(tokenized_dataset_eval, batch_size=1, collate_fn=data_collator, shuffle=False)
    accelerator = Accelerator()
    dataloader = accelerator.prepare(dataloader)
    num_eval_steps = len(dataloader)
    print(tokenizer.decode([0,1,2,3]))
    for step, batch in tqdm(enumerate(dataloader), total=num_eval_steps):
        input_ids = batch['input_ids'][..., :128]
        labels = batch['labels'][..., :128]
        terminate = False
        if labels[0][-1] == -100: continue
        print(tokenizer.decode(input_ids[0]))