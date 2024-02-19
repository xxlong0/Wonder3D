CUDA_VISIBLE_DEVICES=0 \
 blenderproc run --blender-install-path /mnt/pfs/users/longxiaoxiao/workplace/blender \
 blenderProc_nineviews_ortho.py \
 --object_path /mnt/pfs/data/objaverse_lvis_glbs/c7/c70e8817b5a945aca8bb37e02ddbc6f9.glb --view 0 \
 --output_folder ./out_renderings/ \
 --object_uid c70e8817b5a945aca8bb37e02ddbc6f9 \
 --ortho_scale 1.35 \
 --resolution 512 \
#  --reset_object_euler