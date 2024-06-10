export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
export OMP_NUM_THREADS=48
torchrun --nproc_per_node=6 tests/pipetree_benchmark.py --target_group 0 1 2 3 --draft_group 4 5 --model meta-llama/Llama-2-7b-hf --target meta-llama/Llama-2-70b-hf --T 0.6 --P 0.9 --M 256 --B 16 --growmap demo_tree.pt --Mode fast
# torchrun --nproc_per_node=3 tests/pipetree_benchmark.py --target_group 0 1 --draft_group 2 --model princeton-nlp/Sheared-LLaMA-1.3B --target meta-llama/Llama-2-7b-hf --T 0.6 --P 0.9 --M 256 --B 16 --growmap demo_tree.pt