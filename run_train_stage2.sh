# stage 2
accelerate launch --config_file 1gpu.yaml train_mvdiffusion_joint.py --config configs/train/stage2-joint-6views-lvis.yaml