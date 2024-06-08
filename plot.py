import argparse
import time
import torch
import sys
sys.path.append("..")
import torch.distributed as dist
import numpy as np
import os
import torch.distributed as dist
from Hesse.Engine.utils import initialized_dist_baseline, args_parse_baseline, make_causal_mask
from Hesse.Engine.pipleline import LLM_Pipeline
import matplotlib.pyplot as plt



args=args_parse_baseline()
pp_config=initialized_dist_baseline(args.tp_groups,args.layer_partition)
print(args)
print("="*80)
print(pp_config)
global_rank=dist.get_rank()

MAX_LEN = args.M
DEC_LEN = args.D
MODEL_NAME = args.model
DTYPE = torch.float16
# DEVICE = torch.device("cuda", 0)
DEVICE = torch.device("cuda", global_rank)
PREFIX_LEN= args.P
T = 100
WARM_UP = 10
latency_list=[]
bsz_list = [1]
for batch_size in bsz_list:
    engine = LLM_Pipeline(max_length=MAX_LEN, model_name=args.model, device=DEVICE, pp_config=pp_config, batch_size=batch_size, type = "baseline", cg_list=[1])
    input_ids = torch.randint(low=3, high=30000, size=(batch_size, PREFIX_LEN), device=DEVICE)
    attention_mask = make_causal_mask((MAX_LEN, MAX_LEN), dtype=DTYPE, device=DEVICE)
    attention_mask = attention_mask[None, None, :, :]
    position_ids = torch.arange(PREFIX_LEN, device=DEVICE).repeat(batch_size, 1)
    prefix_storage_ids = torch.arange(PREFIX_LEN, device=DEVICE)
    logits = engine.forward(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask[..., :PREFIX_LEN,:], storage_ids=prefix_storage_ids)

    input_ids = torch.randint(low=3, high=30000, size=(batch_size, DEC_LEN), device=DEVICE)
    storage_ids = torch.arange(DEC_LEN, device=DEVICE) + PREFIX_LEN
    position_ids = storage_ids.clone().repeat(batch_size, 1)
    attention_mask = attention_mask[..., PREFIX_LEN: PREFIX_LEN + DEC_LEN,:].clone()

    for _ in range(WARM_UP):
        engine.forward(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask, storage_ids=storage_ids)

    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(T):
        engine.forward(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask, storage_ids=storage_ids)
    torch.cuda.synchronize()
    t2 = time.time()
    if dist.get_rank() == 0:
        print("Max Length :{}, Decode Length :{}, Prefix Length :{}, inference time:{}s".format(MAX_LEN, DEC_LEN, PREFIX_LEN, (t2 - t1)/ T))
    engine = None
    latency_list.append((t2 - t1)/ T)

if global_rank == 0:
    throughput = [1 / latency_val for latency_val in latency_list]
    result = [batch * lat for batch, lat in zip(bsz_list, throughput)]
    plt.plot(bsz_list, result, marker='o')
    plt.xlabel('Batch Size')
    plt.ylabel('Throughput')
    plt.title('Throughput vs. Batch Size')
    plt.savefig("plot.jpg")
