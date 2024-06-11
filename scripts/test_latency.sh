export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_P2P_DISABLE=0
export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=48

torchrun --nproc_per_node=4 --master_port=13456 tests/test_latency.py --B 1 --model meta-llama/Llama-2-70b-hf --M 288 --D 2 4 8 16 32 64 --P 128
torchrun --nproc_per_node=4 --master_port=13456 tests/test_latency.py --B 64 --model meta-llama/Llama-2-70b-hf --M 288 --D 2 4 8 16 32 64 --P 128
torchrun --nproc_per_node=5 --master_port=13456 tests/test_latency.py --B 64 --model meta-llama/Llama-2-70b-hf --M 288 --D 2 4 8 16 32 64 --P 128
torchrun --nproc_per_node=6 --master_port=13456 tests/test_latency.py --B 64 --model meta-llama/Llama-2-70b-hf --M 288 --D 2 4 8 16 32 64 --P 128
torchrun --nproc_per_node=7 --master_port=13456 tests/test_latency.py --B 64 --model meta-llama/Llama-2-70b-hf --M 288 --D 2 4 8 16 32 64 --P 128
torchrun --nproc_per_node=4 --master_port=13456 tests/test_latency.py --B 64 --model meta-llama/Llama-2-70b-hf --M 288 --D 2 4 8 16 32 64 --P 128


# torchrun --nproc_per_node=8 --master_port=13456 tests/test_latency.py --B 2 --model meta-llama/Llama-2-70b-hf --M 288 --D 2 4 8 16 32 64 --P 128
# torchrun --nproc_per_node=8 --master_port=13456 tests/test_latency.py --B 4 --model meta-llama/Llama-2-70b-hf --M 288 --D 2 4 8 16 32 64 --P 128
# torchrun --nproc_per_node=8 --master_port=13456 tests/test_latency.py --B 8 --model meta-llama/Llama-2-70b-hf --M 288 --D 2 4 8 16 32 64 --P 128
# torchrun --nproc_per_node=8 --master_port=13456 tests/test_latency.py --B 16 --model meta-llama/Llama-2-70b-hf --M 288 --D 2 4 8 16 32 64 --P 128
# torchrun --nproc_per_node=8 --master_port=13456 tests/test_latency.py --B 32 --model meta-llama/Llama-2-70b-hf --M 288 --D 2 4 8 16 32 64 --P 128
# torchrun --nproc_per_node=8 --master_port=13456 tests/test_latency.py --B 64 --model meta-llama/Llama-2-70b-hf --M 288 --D 2 4 8 16 32 64 --P 128
# torchrun --nproc_per_node=8 --master_port=13456 tests/test_latency.py --B 128 --model meta-llama/Llama-2-70b-hf --M 288 --D 2 4 8 16 32 64 --P 128

# torchrun --nproc_per_node=4 --master_port=13456 tests/test_latency.py --B 1 --model meta-llama/Llama-2-70b-hf --M 288 --D 2 4 8 16 32 64 --P 128
# torchrun --nproc_per_node=4 --master_port=13456 tests/test_latency.py --B 2 --model meta-llama/Llama-2-70b-hf --M 288 --D 2 4 8 16 32 64 --P 128
# torchrun --nproc_per_node=4 --master_port=13456 tests/test_latency.py --B 4 --model meta-llama/Llama-2-70b-hf --M 288 --D 2 4 8 16 32 64 --P 128
# torchrun --nproc_per_node=4 --master_port=13456 tests/test_latency.py --B 8 --model meta-llama/Llama-2-70b-hf --M 288 --D 2 4 8 16 32 64 --P 128
# torchrun --nproc_per_node=4 --master_port=13456 tests/test_latency.py --B 16 --model meta-llama/Llama-2-70b-hf --M 288 --D 2 4 8 16 32 64 --P 128
# torchrun --nproc_per_node=4 --master_port=13456 tests/test_latency.py --B 32 --model meta-llama/Llama-2-70b-hf --M 288 --D 2 4 8 16 32 64 --P 128
# torchrun --nproc_per_node=4 --master_port=13456 tests/test_latency.py --B 64 --model meta-llama/Llama-2-70b-hf --M 288 --D 2 4 8 16 32 64 --P 128

# torchrun --nproc_per_node=5 --master_port=13456 tests/test_latency.py --B 1 --model meta-llama/Llama-2-70b-hf --M 288 --D 2 4 8 16 32 64 --P 128
# torchrun --nproc_per_node=5 --master_port=13456 tests/test_latency.py --B 2 --model meta-llama/Llama-2-70b-hf --M 288 --D 2 4 8 16 32 64 --P 128
# torchrun --nproc_per_node=5 --master_port=13456 tests/test_latency.py --B 4 --model meta-llama/Llama-2-70b-hf --M 288 --D 2 4 8 16 32 64 --P 128
# torchrun --nproc_per_node=4 --master_port=13456 tests/test_latency.py --B 8 --model meta-llama/Llama-2-70b-hf --M 288 --D 2 4 8 16 32 64 --P 128
# torchrun --nproc_per_node=5 --master_port=13456 tests/test_latency.py --B 16 --model meta-llama/Llama-2-70b-hf --M 288 --D 2 4 8 16 32 64 --P 128
# torchrun --nproc_per_node=5 --master_port=13456 tests/test_latency.py --B 32 --model meta-llama/Llama-2-70b-hf --M 288 --D 2 4 8 16 32 64 --P 128
# torchrun --nproc_per_node=5 --master_port=13456 tests/test_latency.py --B 64 --model meta-llama/Llama-2-70b-hf --M 288 --D 2 4 8 16 32 64 --P 128

# torchrun --nproc_per_node=6 --master_port=13456 tests/test_latency.py --B 1 --model meta-llama/Llama-2-70b-hf --M 288 --D 2 4 8 16 32 64 --P 128
# torchrun --nproc_per_node=6 --master_port=13456 tests/test_latency.py --B 2 --model meta-llama/Llama-2-70b-hf --M 288 --D 2 4 8 16 32 64 --P 128
# torchrun --nproc_per_node=6 --master_port=13456 tests/test_latency.py --B 4 --model meta-llama/Llama-2-70b-hf --M 288 --D 2 4 8 16 32 64 --P 128
# torchrun --nproc_per_node=6 --master_port=13456 tests/test_latency.py --B 8 --model meta-llama/Llama-2-70b-hf --M 288 --D 2 4 8 16 32 64 --P 128
# torchrun --nproc_per_node=6 --master_port=13456 tests/test_latency.py --B 16 --model meta-llama/Llama-2-70b-hf --M 288 --D 2 4 8 16 32 64 --P 128
# torchrun --nproc_per_node=6 --master_port=13456 tests/test_latency.py --B 32 --model meta-llama/Llama-2-70b-hf --M 288 --D 2 4 8 16 32 64 --P 128
# torchrun --nproc_per_node=6 --master_port=13456 tests/test_latency.py --B 64 --model meta-llama/Llama-2-70b-hf --M 288 --D 2 4 8 16 32 64 --P 128

# torchrun --nproc_per_node=7 --master_port=13456 tests/test_latency.py --B 1 --model meta-llama/Llama-2-70b-hf --M 288 --D 2 4 8 16 32 64 --P 128
# torchrun --nproc_per_node=7 --master_port=13456 tests/test_latency.py --B 2 --model meta-llama/Llama-2-70b-hf --M 288 --D 2 4 8 16 32 64 --P 128
# torchrun --nproc_per_node=7 --master_port=13456 tests/test_latency.py --B 4 --model meta-llama/Llama-2-70b-hf --M 288 --D 2 4 8 16 32 64 --P 128
# torchrun --nproc_per_node=6 --master_port=13456 tests/test_latency.py --B 8 --model meta-llama/Llama-2-70b-hf --M 288 --D 2 4 8 16 32 64 --P 128
# torchrun --nproc_per_node=7 --master_port=13456 tests/test_latency.py --B 16 --model meta-llama/Llama-2-70b-hf --M 288 --D 2 4 8 16 32 64 --P 128
# torchrun --nproc_per_node=7 --master_port=13456 tests/test_latency.py --B 32 --model meta-llama/Llama-2-70b-hf --M 288 --D 2 4 8 16 32 64 --P 128
# torchrun --nproc_per_node=7 --master_port=13456 tests/test_latency.py --B 64 --model meta-llama/Llama-2-70b-hf --M 288 --D 2 4 8 16 32 64 --P 128