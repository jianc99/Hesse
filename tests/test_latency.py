import sys
sys.path.append("..")
from Hesse.Engine.llm_pipe import LLMEngine
from Hesse.Engine.utils import initialized_dist_baseline, make_causal_mask, setup_seed
import argparse
import time
import torch
import torch.distributed as dist

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="meta-llama/Llama-2-7b-hf",help='model')
parser.add_argument('--T', type=int, default=500, help='repeat times')
parser.add_argument('--B', type=int, default=1, help='batch size')
parser.add_argument('--P', type=int, default=128, help='prefix length')
parser.add_argument('--M', type=int, default=288, help='max length')
parser.add_argument('--D', type=int, default=1, help='dec length')
args = parser.parse_args()
print(args)
global_group = initialized_dist_baseline()
setup_seed(123)
local_rank = dist.get_rank()
world_size = dist.get_world_size()
PREFIX_LEN = args.P
MAX_LEN = args.M
DEC_LEN = args.D
MODEL_NAME = args.model
DTYPE = torch.float16
DEVICE = torch.device("cuda", local_rank)
BATCH_SIZE = args.B
T = args.T
WARM_UP = 10

llm = LLMEngine(max_length=MAX_LEN, model_name=args.model, device=DEVICE, batch_size=BATCH_SIZE, global_group=global_group, dtype=DTYPE)
llm.initialize_cuda_graph([DEC_LEN])

input_ids = torch.randint(low=3, high=30000, size=(BATCH_SIZE, PREFIX_LEN), device=DEVICE)
attention_mask = make_causal_mask((MAX_LEN, MAX_LEN), dtype=DTYPE, device=DEVICE)
attention_mask = attention_mask[None, None, :, :].repeat(BATCH_SIZE,1,1,1)
position_ids = torch.arange(PREFIX_LEN, device=DEVICE).repeat(BATCH_SIZE,1)
prefix_storage_ids = torch.arange(PREFIX_LEN, device=DEVICE)
llm.forward(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask[..., :PREFIX_LEN,:], storage_ids=prefix_storage_ids)

input_ids = torch.randint(low=3, high=30000, size=(BATCH_SIZE, DEC_LEN), device=DEVICE)
storage_ids = torch.arange(DEC_LEN, device=DEVICE) + PREFIX_LEN
position_ids = storage_ids.clone().repeat(BATCH_SIZE,1)
attention_mask_decode = attention_mask[..., PREFIX_LEN: PREFIX_LEN + DEC_LEN,:].clone()
for _ in range(WARM_UP):
    llm.forward(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask_decode, storage_ids=storage_ids)

torch.cuda.synchronize()
t1 = time.time()
for _ in range(T):
    llm.forward(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask_decode, storage_ids=storage_ids)
torch.cuda.synchronize()
t2 = time.time()
if local_rank == 0:
    print("Batch Size,{}, Max Length :{}, Decode Length :{}, Prefix Length :{}, inference time:{}s".format(BATCH_SIZE, MAX_LEN, DEC_LEN, PREFIX_LEN, (t2 - t1)/ T))



