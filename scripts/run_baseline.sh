export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
export OMP_NUM_THREADS=48
torchrun --nproc_per_node=6 --master_port=13456 tests/baseline_benchmark.py --B 16 --model meta-llama/Llama-2-70b-hf