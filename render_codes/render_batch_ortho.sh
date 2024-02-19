 python distributed.py \
	--num_gpus 8 --gpu_list 0 1 2 3 4 5 6 7 --mode render_ortho    \
	--workers_per_gpu 10 --view_idx $1 \
	--start_i $2 --end_i $3 --ortho_scale 1.35 \
	--input_models_path ../data_lists/lvis_uids_filter_by_vertex.json  \
	--objaverse_root /data/objaverse \
	--save_folder data/obj_lvis_13views \
	--blender_install_path /workplace/blender \
	--random_pose
