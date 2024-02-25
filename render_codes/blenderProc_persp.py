import blenderproc as bproc

import argparse, sys, os, math, re
import bpy
from glob import glob
from mathutils import Vector, Matrix
import random
import sys
import time
import urllib.request
from typing import Tuple
import numpy as np
from blenderproc.python.types.MeshObjectUtility import MeshObject, convert_to_meshes
import pdb

from math import radians
import cv2
from scipy.spatial.transform import Rotation as R

import PIL.Image as Image

parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
parser.add_argument('--view', type=int, default=0,
                    help='the index of view to be rendered')
parser.add_argument(
    "--object_path",
    type=str,
    default='/ghome/l5/xxlong/.objaverse/hf-objaverse-v1/glbs/000-148/2651a32fb4dc441dab773b8b534b851f.glb',
    required=True,
    help="Path to the object file",
)
parser.add_argument('--output_folder', type=str, default='output',
                    help='The path the output will be dumped to.')
parser.add_argument('--resolution', type=int, default=256,
                    help='Resolution of the images.')
parser.add_argument('--object_uid', type=str, default=None)

parser.add_argument('--random_pose', action='store_true',
                    help='whether randomly rotate the poses to be rendered')
parser.add_argument('--reset_object_euler', action='store_true',
                    help='set object rotation euler to 0')   
 
parser.add_argument('--radius', type=float, default=1.5,
                    help='radius of rendering sphere')
parser.add_argument('--delta_z', type=float, default=0,
                    help='delta_z to rotate poses')
parser.add_argument('--delta_x', type=float, default=0,
                    help='delta_x to rotate poses')
parser.add_argument('--delta_y', type=float, default=0)

# argv = sys.argv[sys.argv.index("--") + 1:]
args = parser.parse_args()

def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)


def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj

def normalize_scene():
    bbox_min, bbox_max = scene_bbox()

    dxyz = bbox_max - bbox_min
#    dist = np.sqrt(dxyz[0]**2+ dxyz[1]**2+dxyz[2]**2)
#    print("dxyz: ",dxyz, "dist: ", dist)
    scale = 1 / max(bbox_max - bbox_min)
 #   scale = 1. / dist
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale
    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")

    return scale, offset

def get_a_camera_location(loc):
    location = Vector([loc[0],loc[1],loc[2]])
    direction = - location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    rotation_euler = rot_quat.to_euler()
    return location, rotation_euler


# function from https://github.com/panmari/stanford-shapenet-renderer/blob/master/render_blender.py
def get_3x4_RT_matrix_from_blender(cam):
    # bcam stands for blender camera
    # R_bcam2cv = Matrix(
    #     ((1, 0,  0),
    #     (0, 1, 0),
    #     (0, 0, 1)))

    # Transpose since the rotation is object rotation, 
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam @ location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam @ cam.location
    # Use location from matrix_world to account for constraints:     
    T_world2bcam = -1*R_world2bcam @ location

    # # Build the coordinate transform matrix from world to computer vision camera
    # R_world2cv = R_bcam2cv@R_world2bcam
    # T_world2cv = R_bcam2cv@T_world2bcam

    # put into 3x4 matrix
    RT = Matrix((
        R_world2bcam[0][:] + (T_world2bcam[0],),
        R_world2bcam[1][:] + (T_world2bcam[1],),
        R_world2bcam[2][:] + (T_world2bcam[2],)
        ))
    return RT

def get_calibration_matrix_K_from_blender(mode='simple'):

    scene = bpy.context.scene

    scale = scene.render.resolution_percentage / 100
    width = scene.render.resolution_x * scale # px
    height = scene.render.resolution_y * scale # px

    camdata = scene.camera.data

    if mode == 'simple':

        aspect_ratio = width / height
        K = np.zeros((3,3), dtype=np.float32)
        K[0][0] = width / 2 / np.tan(camdata.angle / 2)
        K[1][1] = height / 2. / np.tan(camdata.angle / 2) * aspect_ratio
        K[0][2] = width / 2.
        K[1][2] = height / 2.
        K[2][2] = 1.
        K.transpose()
    
    if mode == 'complete':

        focal = camdata.lens # mm
        sensor_width = camdata.sensor_width # mm
        sensor_height = camdata.sensor_height # mm
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y

        if (camdata.sensor_fit == 'VERTICAL'):
            # the sensor height is fixed (sensor fit is horizontal), 
            # the sensor width is effectively changed with the pixel aspect ratio
            s_u = width / sensor_width / pixel_aspect_ratio 
            s_v = height / sensor_height
        else: # 'HORIZONTAL' and 'AUTO'
            # the sensor width is fixed (sensor fit is horizontal), 
            # the sensor height is effectively changed with the pixel aspect ratio
            pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
            s_u = width / sensor_width
            s_v = height * pixel_aspect_ratio / sensor_height

        # parameters of intrinsic calibration matrix K
        alpha_u = focal * s_u
        alpha_v = focal * s_v
        u_0 = width / 2
        v_0 = height / 2
        skew = 0 # only use rectangular pixels

        K = np.array([
            [alpha_u,    skew, u_0],
            [      0, alpha_v, v_0],
            [      0,       0,   1]
        ], dtype=np.float32)
    
    return K

# load the glb model
def load_object(object_path: str) -> None:
    """Loads a glb model into the scene."""
    if object_path.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=False)
    elif object_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=object_path)
    elif object_path.endswith(".obj"):
        bpy.ops.import_scene.obj(filepath=object_path)
    elif object_path.endswith(".ply"):
        bpy.ops.import_mesh.ply(filepath=object_path)
    else:
        raise ValueError(f"Unsupported file type: {object_path}")

def reset_scene() -> None:
    """Resets the scene to a clean state."""
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)

bproc.init()

world_tree = bpy.context.scene.world.node_tree
back_node = world_tree.nodes['Background']
env_light = 0.5
back_node.inputs['Color'].default_value = Vector([env_light, env_light, env_light, 1.0])
back_node.inputs['Strength'].default_value = 1.0

#Place camera
cam = bpy.context.scene.objects['Camera']
# cam.location = (0, 1, 0.6)
cam.data.lens = 35
cam.data.sensor_width = 32

# cam_constraint = cam.constraints.new(type='TRACK_TO')
# cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
# cam_constraint.up_axis = 'UP_Y'


#Make light just directional, disable shadows.
light = bproc.types.Light(name='Light', light_type='SUN')
light = bpy.data.lights['Light']
light.use_shadow = False
# Possibly disable specular shading:
light.specular_factor = 1.0
light.energy = 5.0

#Add another light source so stuff facing away from light is not completely dark
light2 = bproc.types.Light(name='Light2', light_type='SUN')
light2 = bpy.data.lights['Light2']
light2.use_shadow = False
light2.specular_factor = 1.0
light2.energy = 1 #0.015
bpy.data.objects['Light2'].rotation_euler = bpy.data.objects['Light'].rotation_euler
bpy.data.objects['Light2'].rotation_euler[0] += 180



# Get all camera objects in the scene
def get_camera_objects():
    cameras = [obj for obj in bpy.context.scene.objects if obj.type == 'CAMERA']
    return cameras


VIEWS = ["_front", "_back", "_right", "_left", "_front_right", "_front_left", "_back_right", "_back_left", "_top"]

def save_images(object_file: str, viewidx: int) -> None:
    
    reset_scene()

    # load the object
    load_object(object_file)
    if args.object_uid is None:
        object_uid = os.path.basename(object_file).split(".")[0]
    else:
        object_uid = args.object_uid

    # cname = "_r%.2f_dx%.1f_dy%.1f_dz%.1f" % (args.radius, args.delta_x, args.delta_y, args.delta_z)
    # object_uid = object_uid + cname
    os.makedirs(os.path.join(args.output_folder, object_uid), exist_ok=True)

    if args.reset_object_euler:
        for obj in scene_root_objects():
            obj.rotation_euler[0] = 0  # don't know why
        bpy.ops.object.select_all(action="DESELECT")

    scale , offset = normalize_scene()

    Scale_path = os.path.join(args.output_folder, object_uid, "scale_offset_matrix.txt")
    # print(scale)
    # print(offset)
    np.savetxt(Scale_path, [scale]+list(offset)+[args.delta_x, args.delta_y, args.delta_z])

    try:
        # some objects' normals are affected by textures
        mesh_objects = convert_to_meshes([obj for obj in scene_meshes()])
        for obj in mesh_objects:
            print("removing invalid normals")
            for mat in obj.get_materials():
                mat.set_principled_shader_value("Normal", [1,1,1])
    except:
        print("don't know why")
    
    cam_empty = bpy.data.objects.new("Empty", None)
    cam_empty.location = (0, 0, 0)
    bpy.context.scene.collection.objects.link(cam_empty)
    
    radius = args.radius
    
    camera_locations = [
        np.array([0,-radius,0]),  # camera_front
        np.array([0,radius,0]),  # camera back
        np.array([radius,0,0]),  # camera right
        np.array([-radius,0,0]), # camera left
        np.array([radius,-radius,0]) / np.sqrt(2.) ,  # camera_front_right
        np.array([-radius,-radius,0]) / np.sqrt(2.),  # camera front left
        np.array([radius,radius,0]) / np.sqrt(2.),  # camera back right
        np.array([-radius,radius,0]) / np.sqrt(2.),  # camera back left
        np.array([0,0,radius]),  # camera top
        ] 

    for location in camera_locations:
        _location,_rotation = get_a_camera_location(location)
        bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=_location, rotation=_rotation,scale=(1, 1, 1))
        _camera = bpy.context.selected_objects[0]
        _constraint = _camera.constraints.new(type='TRACK_TO')
        _constraint.track_axis = 'TRACK_NEGATIVE_Z'
        _constraint.up_axis = 'UP_Y'
        _camera.parent = cam_empty
        _constraint.target = cam_empty
        _constraint.owner_space = 'LOCAL'

    bpy.context.view_layer.update()

    bpy.ops.object.select_all(action='DESELECT')
    cam_empty.select_set(True)
    
    if args.random_pose:
        print("random poses")
        delta_z = np.random.uniform(-60, 60, 1)  # left right rotate
        delta_x = np.random.uniform(-15, 30, 1)  # up and down rotate
        delta_y = 0
    else:
        print("fix poses")
        delta_z = args.delta_z
        delta_x = args.delta_x
        delta_y = args.delta_y
        

    bpy.ops.transform.rotate(value=math.radians(delta_z),orient_axis='Z',orient_type='VIEW')
    bpy.ops.transform.rotate(value=math.radians(delta_y),orient_axis='Y',orient_type='VIEW')
    bpy.ops.transform.rotate(value=math.radians(delta_x),orient_axis='X',orient_type='VIEW')
    
    bpy.ops.object.select_all(action='DESELECT')
    
            
    for j in range(9):
        view = f"{viewidx:03d}"+ VIEWS[j]
        # set camera
        cam = bpy.data.objects[f'Camera.{j+1:03d}']
        location, rotation = cam.matrix_world.decompose()[0:2]
        
        print(j, rotation)
        
        cam_pose = bproc.math.build_transformation_mat(location, rotation.to_matrix())
        bproc.camera.set_resolution(args.resolution, args.resolution)
        bproc.camera.add_camera_pose(cam_pose)

        # save camera RT matrix
        RT = get_3x4_RT_matrix_from_blender(cam)
        # print(np.linalg.inv(cam_pose))  # the same
        # print(RT)
        # idx = 4*i+j
        RT_path = os.path.join(args.output_folder, object_uid, view+"_RT.txt")
        K_path = os.path.join(args.output_folder, object_uid, view+"_K.txt")
        # NT_path = os.path.join(args.output_folder, object_uid, f"{i:03d}_NT.npy")
        K = get_calibration_matrix_K_from_blender()
        np.savetxt(RT_path, RT)
        np.savetxt(K_path, K)

    
    # activate normal and depth rendering
    # must be here
    bproc.renderer.enable_normals_output()
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    # Render the scene
    data = bproc.renderer.render()

    for j in range(9):
        index = j
        
        view = f"{viewidx:03d}"+ VIEWS[j]
        
        # Nomralizes depth maps
        depth_map = data['depth'][index]
        depth_max = np.max(depth_map)
        valid_mask = depth_map!=depth_max
        invalid_mask = depth_map==depth_max
        depth_map[invalid_mask] = 0
        
        depth_map = np.uint16((depth_map / 10) * 65535)

        normal_map = data['normals'][index]*255

        valid_mask = valid_mask.astype(np.int8)*255

        color_map = data['colors'][index]
        color_map = np.concatenate([color_map, valid_mask[:, :, None]], axis=-1)

        Image.fromarray(color_map.astype(np.uint8)).save(
        '{}/{}/rgb_{}.webp'.format(args.output_folder, object_uid, view), "webp", quality=100)
        
        Image.fromarray(normal_map.astype(np.uint8)).save(
        '{}/{}/normals_{}.webp'.format(args.output_folder, object_uid, view), "webp", quality=100)
        
        # cv2.imwrite('{}/{}/rgb_{}.png'.format(args.output_folder, object_uid, view), color_map)
        # cv2.imwrite('{}/{}/depth_{}.png'.format(args.output_folder,object_uid, view), depth_map)
        # cv2.imwrite('{}/{}/normals_{}.png'.format(args.output_folder,object_uid, view), normal_map)
        # cv2.imwrite('{}/{}/mask_{}.png'.format(args.output_folder,object_uid, view), valid_mask)


def download_object(object_url: str) -> str:
    """Download the object and return the path."""
    # uid = uuid.uuid4()
    uid = object_url.split("/")[-1].split(".")[0]
    tmp_local_path = os.path.join("tmp-objects", f"{uid}.glb" + ".tmp")
    local_path = os.path.join("tmp-objects", f"{uid}.glb")
    # wget the file and put it in local_path
    os.makedirs(os.path.dirname(tmp_local_path), exist_ok=True)
    urllib.request.urlretrieve(object_url, tmp_local_path)
    os.rename(tmp_local_path, local_path)
    # get the absolute path
    local_path = os.path.abspath(local_path)
    return local_path

if __name__ == "__main__":
    # try:
    start_i = time.time()
    if args.object_path.startswith("http"):
        local_path = download_object(args.object_path)
    else:
        local_path = args.object_path
        
    if not os.path.exists(local_path):
        print("object does not exists")
    else:
        try:
            save_images(local_path, args.view)
        except Exception as e:
            print("Failed to render", args.object_path)
            print(e)
        
    end_i = time.time()
    print("Finished", local_path, "in", end_i - start_i, "seconds")
    # delete the object if it was downloaded
    if args.object_path.startswith("http"):
        os.remove(local_path)
    # except Exception as e:
    #     print("Failed to render", args.object_path)
    #     print(e)
