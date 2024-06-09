export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
export OMP_NUM_THREADS=48
torchrun --nproc_per_node=7 --master_port=13456 tests/baseline_benchmark.py --B 1 --model meta-llama/Llama-2-70b-hf