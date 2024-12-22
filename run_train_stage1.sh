export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1

accelerate launch --config_file 8gpu.yaml train_mvdiffusion_mixed.py --config configs/train/stage1-mixed-wo-switcher.yaml