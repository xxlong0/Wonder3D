export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=info
export NCCL_SOCKET_IFNAME=eth0
export NCCL_P2P_LEVEL=NVL

accelerate launch --config_file 8gpu.yaml train_mvdiffusion_mixed.py --config configs/train/stage1-mixed-6views-image-normal.yaml
