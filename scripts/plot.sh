export CUDA_VISIBLE_DEVICES=3
export OMP_NUM_THREADS=48
torchrun --nproc_per_node=1 tests/plot.py --layer_partition 32 --tp_groups 1 --model meta-llama/Llama-2-7b-hf