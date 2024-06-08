export CUDA_VISIBLE_DEVICES=3,4,5,6
export OMP_NUM_THREADS=48
torchrun --nproc_per_node=4 tests/seq_spec_benchmark.py --target_layer_partition 80 --target_tp_groups 4 --target_group 0 1 2 3 --draft_layer_partition 32 --draft_tp_groups 4 --draft_group 0 1 2 3