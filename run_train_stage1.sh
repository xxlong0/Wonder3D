
# stage 1
accelerate launch --config_file 1gpu.yaml train_mvdiffusion_image.py --config configs/train/stage1-mix-6views-lvis.yaml