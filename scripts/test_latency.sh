export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=48
torchrun --nproc_per_node=4 --master_port=13456 tests/test_latency.py --B 24 --model meta-llama/Llama-2-7b-hf --M 576 --D 320 --P 128