# Prepare the rendering data

## Environment
The rendering codes are mainly based on [BlenderProc](https://github.com/DLR-RM/BlenderProc). Thanks for the great tool.
BlenderProc uses blender Cycle engine to render the images by default, which may meet long-time hanging problem in some specific GPUs (like A800, tested already)

`
cd ./render_codes
pip install -r requirements.txt
`

## How to use the code
Here we provide two rendering scripts `blenderProc_ortho.py` and `blenderProc_persp.py`, which use **orthogonal** camera and **perspective** camera to render the objects respectively. 

### Use `blenderProc_ortho.py` to render images of a single object
`
 blenderproc run --blender-install-path /mnt/pfs/users/longxiaoxiao/workplace/blender 
 blenderProc_ortho.py 
 --object_path /mnt/pfs/data/objaverse_lvis_glbs/c7/c70e8817b5a945aca8bb37e02ddbc6f9.glb --view 0 
 --output_folder ./out_renderings/ 
 --object_uid c70e8817b5a945aca8bb37e02ddbc6f9 
 --ortho_scale 1.35 
 --resolution 512 
 --random_pose
`

Here `--view` denotes a tag for the rendering images, since you may render an object multiple times, `--ortho_scale` decides the scaling of rendered object in the image, `--random_pose` will randomly rotate the object before rendering.


### Use `blenderProc_persp.py` to render images of a single object

`
 blenderproc run --blender-install-path /mnt/pfs/users/longxiaoxiao/workplace/blender 
 blenderProc_persp.py 
 --object_path ${the object path} --view 0 
 --output_folder ${your save path}
 --object_uid ${object_uid} --radius 2.0 
 --random_pose
`

Here `--radius` denotes the distance of between the camera and the object origin.

### Render objects in distributed mode
see `render_batch_ortho.sh` and `render_batch_persp.sh` for commands.

