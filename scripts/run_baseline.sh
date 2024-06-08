export CUDA_VISIBLE_DEVICES=0,1,2
export OMP_NUM_THREADS=48
torchrun --nproc_per_node=3 --master_port=13456 tests/baseline_benchmark.py --layer_partition 32 --tp_groups 3