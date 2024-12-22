 python distributed.py \
	--num_gpus 8 --gpu_list 0 1 2 3 4 5 6 7 --mode render_ortho    \
	--workers_per_gpu 7 --view_idx 0 \
	--start_i 0 --end_i -1 --ortho_scale 1.35 \
	--input_models_path ../datalist/has_uvs.json  \
	--objaverse_root ~/.objaverse/hf-objaverse-v1 \
	--save_folder data/13views \
	--blender_install_path /root/workspace/blender/blender-3.3.0-linux-x64 \
	--random_pose