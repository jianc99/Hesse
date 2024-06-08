# Hesse
## Installation
### Create Virtual Environment
``` bash
conda create -n hesse python=3.11
conda activate hesse
```

### Install Necessary Packages
Must ensure NCCL version to be the same across different nodes. In our experiments, NCCL version=2.18.6.

``` bash
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.36.2
pip install protobuf
pip install sentencepiece
pip install datasets==2.16.1
pip install accelerate==0.26.1
pip install matplotlib
```

## Run Scripts
Run the scripts for each GPU worker. Need to specify the master address and port, and NCCL_SOCKET_IFNAME to specific network interface.
``` bash
export MASTER_ADDR='172.19.136.149'
export MASTER_PORT=9991
export WORLD_SIZE=6
export RANK=0
export NCCL_SOCKET_IFNAME=eno1

CUDA_VISIBLE_DEVICES=0 python3 autoregressive_inference.py --layer_partition 10 11 11 --tp_groups 2 2 2
```

``` bash
export MASTER_ADDR='172.19.136.149'
export MASTER_PORT=9991
export WORLD_SIZE=8
export RANK=0
export NCCL_SOCKET_IFNAME=eno1

CUDA_VISIBLE_DEVICES=0 python3 speculative_decoding.py --target_layer_partition 40 40 --target_tp_groups 4 4 --target_group 0 1 2 3 4 5 6 7 --draft_layer_partition 32 --draft_tp_groups 8 --draft_group 0 1 2 3 4 5 6 7
```

To compare the inference result with huggingface model, just run with the same prompt
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python hf_output.py
```


## Performance on A100 80G PCIE
Unit in ms, Prefix = 512, Batch size = 1
| Model / # GPUs | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| Sheared-LLaMA-2.7B  |  7.9 |   |   |  |
| Llama-2-7b  | 12.7  | 10.2  | 8.2  |   |
| Llama-2-13b  | 21.6 |   |   |   |
| Llama-2-70b | x  |   |   |   |
| vicuna-33b-v1.3 | 49.0  |   |   |   |

## Performance on A100 80G SXM
Unit in ms, Prefix = 512, Batch size = 1
| Model / # GPUs | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| Llama-2-70b | x  | 59.0 | 37.5  | 27.7 |

## Performance on H100 80G PCIE
Unit in ms, Prefix = 512, Batch size = 1
| Model / # GPUs | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| Llama-2-7b  | 12.7  | 9.0  | 7.3  |   |

## Performance on 4090 24G PCIE
Unit in ms, Prefix = 512, Batch size = 1
| Model / # GPUs | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| Llama-2-7b  | 17.1  | 11.3  | 7.5  | 5.9  |
| Llama-2-70b | x  |  x | x  | 29.9  |
| vicuna-33b-v1.3 | x  | x  | 25.0  | x  |

## Performance on L40 48G PCIE
Unit in ms, Prefix = 512, Batch size = 1
| Model / # GPUs | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| Llama-2-7b  | 22.1  | 14.4  | 9.0  | 7.0  |
| Llama-2-70b | x  |  x | 69.9  | x  |

PP+TP Degree= 4 4 means the first and second pipeline stages are both doing tensor parallelism with degree=4.

| PP+TP Degree | 2 2 | 2 2 2 | 4 4 |
|---|---|---|---|
| Llama-2-7b  | 14.6  | 14.6 | 9.1 |