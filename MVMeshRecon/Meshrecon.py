import numpy as np
from remeshing.util.render import Renderer
from refine_texture.api import opt_warpper
from utils.generate_sphere import make_sphere
from remeshing.util.func import save_images, save_obj, save_images_init
from remeshing.core.remesh import calc_vertex_normals
import torch
from remeshing.optimize import do_optimize
from MeshRecon.optimize import simple_clean_mesh, to_pyml_mesh, geo_aware_mesh_refine

def MeshRecon(mv_normal, proj_normal, gt_normals, mv_RGB, proj_RGB, gt_RGBs, RGB_refine_index=None, RGB_view_weights=None, vertices_init=None, faces_init=None, debug_path = None, clean_mesh=True, persp=True, output_path='debug'):
    if debug_path is None:
        debug_path = "debug/"
    if RGB_refine_index is None:
        RGB_refine_index = np.arange(len(mv_RGB))
    if RGB_view_weights is None:
        RGB_view_weights = [1]*len(RGB_refine_index) 
    
    renderer = Renderer(mv_normal, proj_normal, [gt_normals.shape[1], gt_normals.shape[2]] )
    if vertices_init is None:
        vertices_init, faces_init = make_sphere(radius=3)
    save_obj(vertices_init, faces_init, "debug/init_3d_model")

    gt_normals_world = gt_normals

    vertices, faces = do_optimize(vertices_init, faces_init, gt_normals_world, renderer, None, edge_len_lims=(0.01, 0.05), remeshing_steps=200)

    vertices, faces = geo_aware_mesh_refine(vertices, faces, gt_normals_world, renderer, mv_normal,
                                                  proj_normal, 100, start_edge_len=0.01,
                                                  end_edge_len=0.008, b_persp=persp, update_normal_interval=10, update_warmup=5)

    meshes = simple_clean_mesh(to_pyml_mesh(vertices, faces), apply_smooth=True, stepsmoothnum=1,
                               apply_sub_divide=True, sub_divide_threshold=0.25).to("cuda")
    vertices = meshes._verts_list[0]
    faces = meshes._faces_list[0]

    render_path = f'{output_path}/render_out'
    torch.cuda.empty_cache()

    textured_mesh = opt_warpper(vertices, faces, gt_RGBs, mv_RGB, proj_RGB, weights=RGB_view_weights, refine_index=RGB_refine_index, render_size=gt_RGBs.shape[-2], b_perspective=persp, visualize=True, do_refine_uv=False)

    renderer._image_size = [512, 512]
    normals = calc_vertex_normals(textured_mesh.mesh.v_pos, textured_mesh.mesh.t_pos_idx)
    normals_rendered = renderer.render_normal(textured_mesh.mesh.v_pos, normals, textured_mesh.mesh.t_pos_idx)
    rgb_rendered = renderer.render_RGB_texture(vertices=textured_mesh.mesh.v_pos,
                                               faces=textured_mesh.mesh.t_pos_idx,
                                               vt=textured_mesh.mesh.v_tex,
                                               ft=textured_mesh.mesh.t_tex_idx,
                                               texture=textured_mesh.map_Kd
                                               )

    rgb_rendered[..., 3] = normals_rendered[..., 3]
    save_images_init(normals_rendered, render_path, name='normal')
    save_images_init(rgb_rendered, render_path, name='rgb')

    return textured_mesh, rgb_rendered, normals_rendered, vertices, faces