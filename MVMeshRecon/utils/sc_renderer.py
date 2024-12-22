import numpy as np
import trimesh, pyrender

def render_glcam(K,
                 Rt,
                 model_in,  # model name or trimesh
                 scale=1.0,
                 std_size=(1000, 1000),
                 flat_shading=False):
    # Mesh creation
    if isinstance(model_in, str) is True:
        mesh = trimesh.load(model_in, process=False)
    else:
        mesh = model_in.copy()
    pr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)

    # Scene creation
    scene = pyrender.Scene(bg_color=(0.0,0.0,0.0,0.0))

    # Adding objects to the scene
    face_node = scene.add(pr_mesh)

    # Caculate fx fy cx cy from K
    fx, fy = K[0][0] * scale, K[1][1] * scale
    cx, cy = K[0][2] * scale, K[1][2] * scale

    # Camera Creation
    cam = pyrender.IntrinsicsCamera(fx, fy, cx, cy,
                                    znear=0.1, zfar=100000)
    cam_pose = np.eye(4)
    cam_pose[:3, :3] = Rt[:3, :3].T
    cam_pose[:3, 3] = -Rt[:3, :3].T.dot(Rt[:, 3])
    scene.add(cam, pose=cam_pose)

    # Set up the light
    # light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=15.0)
    light = pyrender.SpotLight(color=[1.0, 1.0, 1.0], intensity=50.0)
    scene.add(light, pose=cam_pose)

    # Rendering offscreen from that camera
    r = pyrender.OffscreenRenderer(viewport_width=std_size[1],
                                   viewport_height=std_size[0],
                                   point_size=1.0)
    if True:
        color, depth = r.render(scene, flags=pyrender.constants.RenderFlags.RGBA)
    else:
        color = r.render(scene, flags=pyrender.constants.RenderFlags.DEPTH_ONLY)

    # # rgb to bgr for cv2
    # color = color[:, :, [2, 1, 0]]

    return color



# render with cv camera
def render_cvcam(K,
                 Rt,
                 model_in,  # model name or trimesh
                 scale=1.0,
                 std_size=(1000, 1000),
                 flat_shading=False):
    # define R to transform from cvcam to glcam
    R_cv2gl = np.array([[1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1]])
    Rt_cv = R_cv2gl.dot(Rt)

    return render_glcam(K, Rt_cv, model_in, scale, std_size, flat_shading)


# render with gl camera
def render_orthcam(model_in,  # model name or trimesh
                   xy_mag,
                   rend_size,
                   flat_shading=True,
                   zfar=10000,
                   znear=0.05):
    # Mesh creation
    if isinstance(model_in, str) is True:
        mesh = trimesh.load(model_in, process=False)
    else:
        mesh = model_in.copy()
    pr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)

    # Scene creation
    scene = pyrender.Scene()

    # Adding objects to the scene
    face_node = scene.add(pr_mesh)

    # Camera Creation
    cam = pyrender.OrthographicCamera(xmag=xy_mag[0], ymag=xy_mag[1],
                                      znear=znear, zfar=zfar)

    scene.add(cam, pose=np.eye(4))

    # Set up the light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=5.0)
    scene.add(light, pose=np.eye(4))

    # Rendering offscreen from that camera
    r = pyrender.OffscreenRenderer(viewport_width=rend_size[1],
                                   viewport_height=rend_size[0],
                                   point_size=1.0)
    if flat_shading is True:
        depth = r.render(scene, flags=pyrender.constants.RenderFlags.DEPTH_ONLY)
    else:
        depth = r.render(scene, flags=pyrender.constants.RenderFlags.DEPTH_ONLY)

    # rgb to bgr for cv2
    # color = color[:, :, [2, 1, 0]]

    # IMPORTANT! FIX pyrender BUG, pyrender version: 0.1.43
    depth[depth != 0] = (zfar + znear - ((2.0 * znear * zfar) / depth[depth != 0])) / (zfar - znear)
    depth[depth != 0] = ((depth[depth != 0] + (zfar + znear) / (zfar - znear)) * (zfar - znear)) / 2.0

    return depth


def render_orthcam(model,  # model name or trimesh
                   xy_mag,
                   rend_size,
                   flat_shading=True,
                   zfar=10000,
                   znear=0.05):
    # Scene creation
    scene = pyrender.Scene()

    # # for model in model_list:
    # pr_mesh = pyrender.Mesh.from_trimesh(model.copy())
    # Adding objects to the scene
    # Mesh creation
    if isinstance(model, str) is True:
        mesh = trimesh.load(model, process=False)
    else:
        mesh = model.copy()
    pr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)#False)
    face_node = scene.add(pr_mesh)

    # Camera Creation
    cam = pyrender.OrthographicCamera(xmag=xy_mag[0], ymag=xy_mag[1],
                                      znear=znear, zfar=zfar)

    scene.add(cam, pose=np.eye(4))

    # Set up the light
    light = pyrender.DirectionalLight(color=[0.8, 0.8, 0.8], intensity=10.0)#5.0)
    scene.add(light, pose=np.eye(4))

    # Rendering offscreen from that camera
    r = pyrender.OffscreenRenderer(viewport_width=rend_size[1],
                                   viewport_height=rend_size[0],
                                   point_size=1.0)
    if flat_shading is True:
        color, depth = r.render(scene, flags=pyrender.constants.RenderFlags.FLAT)
    else:
        color, depth = r.render(scene)

    # rgb to bgr for cv2
    color = color[:, :, [2, 1, 0]]

    # IMPORTANT! FIX pyrender BUG, pyrender version: 0.1.43
    depth[depth != 0] = (zfar + znear - ((2.0 * znear * zfar) / depth[depth != 0])) / (zfar - znear)
    depth[depth != 0] = ((depth[depth != 0] + (zfar + znear) / (zfar - znear)) * (zfar - znear)) / 2.0

    return depth, color


# render visi map with cv camera (background == np.uint32(-1))
# MUST use 'from opendr.renderer import DepthRenderer'
# or use 'from renderer import *'
# Warning: input mesh should be scaled to near 1-10 level,
#          or some unseen faces may be mis-judged to be visible.
#          This should be caused by bugs in OpenDR
def render_visi_cvcam(K,
                      Rt,
                      model_in,  # model name or trimesh
                      rend_size,  # (h, w)
                      d_max=10000.):
    # mesh creation
    if isinstance(model_in, str) is True:
        mesh = trimesh.load(model_in, process=False)
    else:
        mesh = model_in.copy()

    rn = DepthRenderer()
    rn.camera = ProjectPoints(rt=Rodrigues(Rt[:3, :3]).squeeze(), t=Rt[:3, 3],
                              f=[K[0, 0], K[1, 1]], c=[K[0, 2], K[1, 2]],
                              k=np.zeros(5))
    rn.frustum = {'near': .1, 'far': d_max,
                  'width': rend_size[1], 'height': rend_size[0]}
    rn.v = mesh.vertices
    rn.f = mesh.faces
    rn.bgcolor = np.zeros(3)
    vis_img = rn.visibility_image.copy()
    return vis_img


# verts/faces version of render_visi_cvcam
def render_visi_cvcam_vf(K, Rt, verts, faces,
                         rend_size,  # (h, w)
                         d_max=10000.):
    rn = DepthRenderer()
    rn.camera = ProjectPoints(rt=Rodrigues(Rt[:3, :3]).squeeze(), t=Rt[:3, 3],
                              f=[K[0, 0], K[1, 1]], c=[K[0, 2], K[1, 2]],
                              k=np.zeros(5))
    rn.frustum = {'near': .1, 'far': d_max,
                  'width': rend_size[1], 'height': rend_size[0]}
    rn.v = verts
    rn.f = faces
    rn.bgcolor = np.zeros(3)
    return rn.visibility_image.copy()

