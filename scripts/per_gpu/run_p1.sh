export MASTER_ADDR='172.19.136.149'
export MASTER_PORT=9991
export WORLD_SIZE=6
export RANK=1
export NCCL_SOCKET_IFNAME=eno1
export NCCL_DEBUG=INFO

CUDA_VISIBLE_DEVICES=5 python3 tests/pipeline_benchmark.py --target_layer_partition 80 --target_tp_groups 4 --target_group 0 1 2 3 --draft_layer_partition 32 --draft_tp_groups 2 --draft_group 4 5 --model meta-llama/Llama-2-7b-hf --target meta-llama/Llama-2-70b-hf --T 0.6 --P 0.9 --M 256 --B 16 --growmap 7b-70b_tree.pt