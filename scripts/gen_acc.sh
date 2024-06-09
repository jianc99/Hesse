export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=48
# torchrun --nproc_per_node=8 tests/test_accept.py --target_layer_partition 80 --target_tp_groups 8 --target_group 0 1 2 3 4 5 6 7 --draft_layer_partition 32 --draft_tp_groups 8 --draft_group 0 1 2 3 4 5 6 7 --model meta-llama/Llama-2-7b-hf --target meta-llama/Llama-2-70b-hf --T 0.6 --P 1.0 --M 288 --W 32 --dataset cnn
# torchrun --nproc_per_node=8 tests/test_accept.py --target_layer_partition 80 --target_tp_groups 8 --target_group 0 1 2 3 4 5 6 7 --draft_layer_partition 2 --draft_tp_groups 1 --draft_group 0 --model JackFram/llama-68m --target meta-llama/Llama-2-70b-hf --T 0.6 --P 0.9 --M 288 --W 32 --dataset cnn
# torchrun --nproc_per_node=8 tests/test_accept.py --target_layer_partition 80 --target_tp_groups 8 --target_group 0 1 2 3 4 5 6 7 --draft_layer_partition 24 --draft_tp_groups 2 --draft_group 0 1 --model princeton-nlp/Sheared-LLaMA-1.3B --target meta-llama/Llama-2-70b-hf --T 0.6 --P 0.9 --M 288 --W 32 --dataset cnn
torchrun --nproc_per_node=8 tests/test_accept.py --target_group 0 1 2 3 4 5 6 7 --draft_group 0 1 2 3 4 5 6 7 --model meta-llama/Llama-2-7b-hf --target meta-llama/Llama-2-70b-hf --T 0.6 --P 0.9 --M 256 --W 32 --dataset cnn --dst 7b-70b-acc.pt
# torchrun --nproc_per_node=4 tests/test_accept.py --target_layer_partition 32 --target_tp_groups 4 --target_group 0 1 2 3 --draft_layer_partition 32 --draft_tp_groups 4 --draft_group 0 1 2 3 --model meta-llama/Llama-2-7b-hf --target meta-llama/Llama-2-7b-hf --T 0.6 --P 0 --M 288 --W 32 --dataset cnn

