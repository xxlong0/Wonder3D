import os
import shutil
from typing import Iterable
from tqdm import tqdm
from copy import deepcopy
from glob import glob
import argparse
import json
from omegaconf import OmegaConf
import numpy as np
from PIL import Image
import cv2
import torch
from torchvision.utils import save_image
import trimesh
from trimesh.visual import TextureVisuals
import open3d as o3d
import rembg

from refine_texture.mesh_refine import refine_mesh_uv, refine_mesh_vertex
from refine_texture.common.cameras.mvp_matrix_v2 import proj_to_intr
from refine_texture.common.renderers.nvdiffrast.renderer import NVDiffRendererBase
from refine_texture.common.mesh.structure import Mesh, Texture
from refine_texture.mesh_remapping import remapping_vertex_color, remapping_uv_texture, initial_map_Kd_with_v_rgb
from project_mesh import multiview_color_projection, Meshes, FoVPerspectiveCameras, TexturesVertex
from refine_texture.export_video import VideoExporter
from refine_texture.utils import CPUTimer


def opt_warpper(
    mesh_path, image_path, camera_path, output_path, 
    index_list=None, sr_func=None, 
    initialize_from_vextex=True,
    use_vertex=True,
    use_optimize=True, 
    use_unique_3d=False,
    visualize=False, 
):
    if visualize:
        os.makedirs('./.cache', exist_ok=True)
    
    # build dataset
    dataset_dict = {}
    if camera_path is None:
        from threestudio.data.multiview import MultiviewTestDataset, MultiviewsDataModuleConfig
        mv_test_dataset = MultiviewTestDataset(MultiviewsDataModuleConfig, 'test')
        dataset_dict['frames_c2w'] = mv_test_dataset.c2w
        dataset_dict['intrinsic'] = torch.as_tensor([
            [mv_test_dataset.focal_length[0] / mv_test_dataset.cfg.eval_width, 0, 0.5],
            [0, mv_test_dataset.focal_length[1] / mv_test_dataset.cfg.eval_width, 0.5],
            [0, 0, 1],
        ], dtype=torch.float32).unsqueeze(0).tile(mv_test_dataset.cfg.n_views, 1, 1)
    else:
        json_path = os.path.join(camera_path, 'transforms.json')
        c2w_path = os.path.join(camera_path, 'c2w.npy')
        proj_path = os.path.join(camera_path, 'projection.npy')
        intr_path = os.path.join(camera_path, 'intrinsic.npy')

        if os.path.isfile(json_path):
            with open(json_path, "r") as f:
                data_meta = json.load(f)
            dataset_dict['frames_c2w'] = torch.as_tensor(np.stack(
                [np.asarray(f['transform_matrix']) for f in data_meta['frames']], 
                axis=0,
            ), dtype=torch.float32)
            dataset_dict['intrinsic'] = torch.as_tensor(np.stack([np.asarray([
                [f['fl_x'] / f['w'], 0, f['cx'] / f['w']],
                [0, f['fl_y'] / f['h'], f['cy'] / f['h']],
                [0, 0, 1],
            ]) for f in data_meta['frames']], axis=0), dtype=torch.float32)
        elif os.path.isfile(c2w_path) and (os.path.isfile(proj_path) or os.path.isfile(intr_path)):
            dataset_dict['frames_c2w'] = torch.as_tensor(np.load(c2w_path), dtype=torch.float32)
            if os.path.isfile(proj_path):
                proj_mtx = np.load(proj_path)
                proj_mtx[1, :] = - proj_mtx[1, :]
                dataset_dict['intrinsic'] = proj_to_intr(torch.as_tensor(proj_mtx, dtype=torch.float32).expand(dataset_dict['frames_c2w'].shape[0], 4, 4))
            else:
                dataset_dict['intrinsic'] = torch.as_tensor(np.load(intr_path), dtype=torch.float32).expand(dataset_dict['frames_c2w'].shape[0], 3, 3)         
        else:
            raise NotImplementedError(camera_path)
    
    image_path_list = np.array(sorted(
        glob(os.path.join(image_path, '*.png')), 
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0]),
    ))

    if index_list is not None:
        print(f'remapping frames: {index_list}')
        if isinstance(index_list, str):
            index_list = list(map(int, json.loads(index_list)))
        elif isinstance(index_list, Iterable):
            if len(index_list) == 0:
                raise ValueError('index_list is empty')
        else:
            raise NotImplementedError(index_list)
        image_path_list = image_path_list[index_list]
        dataset_dict['frames_c2w'] = dataset_dict['frames_c2w'][index_list]
        dataset_dict['intrinsic'] = dataset_dict['intrinsic'][index_list]
    else:
        index_list = list(range(len(image_path_list)))
    suffix = f'-{index_list}'.replace(' ', '')
        
    image_list = []
    rembg_session = rembg.new_session()
    for i, path in enumerate(image_path_list):
        im = Image.open(path)
        if im.mode == 'RGBA':
            im = np.array(im, dtype=np.uint8)
            image_list.append(im)
        else:
            im = np.array(im, dtype=np.uint8)
            im = rembg.remove(im, alpha_matting=True, session=rembg_session)
            _im = cv2.inpaint(im[:, :, :3], ~im[:, :, [3]], 3.0, cv2.INPAINT_TELEA)
            im = np.concatenate([_im, im[:, :, [3]]], axis=-1)
            image_list.append(im)
        if visualize:
            Image.fromarray(im, mode='RGBA').save(f'./.cache/source_rgba_{i:04d}.png')
            Image.fromarray(im[..., :3], mode='RGB').save(f'./.cache/source_rgb_{i:04d}.png')
    rgba_list = np.stack(image_list, axis=0)
    # rgba_list = np.concatenate([image_list[:, :512, :512, :3], image_list[:, :512, 1024:, :1]], axis=-1)
    
    if sr_func is not None:
        rgb_sr_list = sr_func(rgba_list[..., :3])
        shutil.copytree("/tmp/figgen/sr", f"{output_path}/sr-temp", dirs_exist_ok=True)
        rgba_list = torch.as_tensor(rgba_list, dtype=torch.float32).div(255).permute(0, 3, 1, 2)
        rgb_sr_list = torch.as_tensor(rgb_sr_list, dtype=torch.float32).div(255).permute(0, 3, 1, 2)
        if rgba_list.shape[2] != rgb_sr_list.shape[2] or rgba_list.shape[3] != rgb_sr_list.shape[3]:
            rgba_list = torch.nn.functional.interpolate(rgba_list, (rgb_sr_list.shape[2], rgb_sr_list.shape[3]), mode='nearest')
        rgba_sr_list = torch.cat([rgb_sr_list, rgba_list[:, [-1], :, :]], dim=1)
        if visualize:
            save_image(torch.cat([rgba_list, rgba_sr_list], dim=3), f"{output_path}/sr-temp-vis/vs.png")
        rgba_list = rgba_sr_list
    else:
        rgba_list = torch.as_tensor(rgba_list, dtype=torch.float32).div(255).permute(0, 3, 1, 2)
    dataset_dict['frames_img'] = rgba_list
    dataset_dict = {k: v.to(device='cuda') if isinstance(v, torch.Tensor) else v for k, v in dataset_dict.items()}

    # load mesh
    # mesh = trimesh.load(mesh_path, process=False)
    mesh = o3d.io.read_triangle_mesh(mesh_path, enable_post_processing=False)
    mesh.remove_non_manifold_edges()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    if not use_vertex:
        if len(mesh.triangles) > 200_000:
            # mesh = mesh.simplify_quadric_decimation(200_000)
            device_o3d = o3d.core.Device('CPU:0')
            mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh, device=device_o3d)
            target_reduction = 1 - 200_000 / len(mesh.triangle.indices)
            mesh = mesh.simplify_quadric_decimation(target_reduction)
            mesh = mesh.to_legacy()
            mesh = mesh.remove_non_manifold_edges()
            mesh = mesh.remove_degenerate_triangles()
            mesh = mesh.remove_unreferenced_vertices()
    else:
        for _ in range(6):
            if len(mesh.triangles) < 800_000:
                mesh = mesh.subdivide_loop()
    mesh = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices, dtype=np.float32),
        faces=np.asarray(mesh.triangles, dtype=np.int64),
        process=False,
    )
    '''mesh.vertices = mesh.vertices @ np.array([
        [0, -1, 0],
        [0, 0, 1],
        [-1, 0, 0],
    ])  # tripo -> nvdiffrast'''
    '''mesh.vertices = mesh.vertices @ np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0],
    ])  # nvdiffrast -> tripo'''
    '''mesh.vertices = mesh.vertices @ np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
    ], dtype=np.float32)  # threestudio -> nvdiffrast'''
    if not use_vertex:
        if not isinstance(mesh.visual, TextureVisuals):
            mesh.visual = TextureVisuals(image=Image.fromarray(np.full((2, 2, 4), fill_value=255, dtype=np.uint8), mode='RGBA'))
        mesh.visual.material.image = mesh.visual.material.image.resize((2048, 2048))
        texture = Texture.from_trimesh(mesh).to(device='cuda')
        texture.reset_map_Kd_mask()
    else:
        mesh.merge_vertices(merge_tex=True, merge_norm=True)
        mesh.update_faces(mesh.nondegenerate_faces())
        texture = Texture.from_trimesh(mesh).to(device='cuda')
    mesh = texture.mesh
    map_Kd = texture.map_Kd
    v_rgb = texture.v_rgb
    # map_Kd[:, :, -1:] = 0.0  # start from zeros
    
    # refine texture
    if not use_vertex and use_optimize:
        mesh_renderer = NVDiffRendererBase()
        mesh, map_Kd = refine_mesh_uv(
            mesh, map_Kd, dataset_dict['frames_c2w'], dataset_dict['intrinsic'], dataset_dict['frames_img'], 
            mesh_renderer.simple_rendering, 
            optimize_extrinsics=False, optimize_intrinsics=False,
            use_vertex_normal=False,
            visualize=visualize,
        )
        v_rgb = None
    if use_vertex and use_optimize:
        mesh_renderer = NVDiffRendererBase()
        mesh, v_rgb = refine_mesh_vertex(
            mesh, map_Kd, dataset_dict['frames_c2w'], dataset_dict['intrinsic'], dataset_dict['frames_img'], 
            mesh_renderer.simple_rendering, 
            optimize_extrinsics=False, optimize_intrinsics=False,
            use_vertex_normal=False,
            visualize=visualize,
        )
        map_Kd = None
    elif use_vertex and not use_optimize:
        if use_unique_3d:
            mesh_pytorch3d = Meshes(
                verts=[mesh.v_pos], 
                faces=[mesh.t_pos_idx], 
                textures=TexturesVertex([v_rgb if v_rgb is not None else torch.zeros_like(mesh.v_pos)])
            )
            image_list = [Image.fromarray(im.mul(255.0).permute(1, 2, 0).to(dtype=torch.uint8).detach().cpu().numpy(), mode='RGB') for im in dataset_dict['frames_img']]
            camera_focal = dataset_dict['intrinsic'][0, 0, 0]
            c2ws = dataset_dict['frames_c2w']
            B, C, H, W = dataset_dict['frames_img'].shape
            c2ws = torch.as_tensor([
                [0, 0, -1, 0],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ], dtype=torch.float32, device=c2ws.device) @ c2ws @ torch.as_tensor([
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ], dtype=torch.float32, device=c2ws.device)  # tripo -> nvdiffrast -> pytorch3d'''
            fov = 2 * torch.atan(1 / (2 * camera_focal))
            cameras_list = [FoVPerspectiveCameras(
                R=c2w[:3, :3].unsqueeze(0), 
                T=c2w[:3, 3].unsqueeze(0), 
                fov=fov, degrees=False,
            ) for c2w in c2ws]
            mesh_pytorch3d = multiview_color_projection(
                mesh_pytorch3d,
                image_list=image_list,
                cameras_list=cameras_list,
                camera_focal=None, 
                weights=None, 
                eps=0.05, 
                resolution=H, 
                device="cuda", 
                reweight_with_cosangle="square", 
                use_alpha=True, 
                confidence_threshold=0.2, 
                complete_unseen=True, 
                below_confidence_strategy="smooth"
            )
            mesh = Mesh(mesh_pytorch3d.verts_list()[0], mesh_pytorch3d.faces_list()[0])
            map_Kd = None
            v_rgb = mesh_pytorch3d.textures.verts_features_list()[0]
        else:
            intrinsics = dataset_dict['intrinsic']
            c2ws = dataset_dict['frames_c2w']
            images = dataset_dict['frames_img']

            # NOTE: hard code, elevation=0; azimuth=0,30,60,...,330
            weights = np.array([
                2.0, 0.05, 0.05, 0.5, 0.05, 0.05,
                1.0, 0.05, 0.05, 0.5, 0.05, 0.05,
            ])
            # stage 1: initialization
            v_rgb = remapping_vertex_color(
                mesh, c2ws, intrinsics, images, v_rgb, 
                weights=[1.0] * len(weights), 
                confidence=0.2,
                use_alpha=False,
                use_inpainting=True,
                visualize=visualize,
            )
            # stage 2: refine from 4 views
            v_rgb = remapping_vertex_color(
                mesh, c2ws[[0, 3, 6, 9]], intrinsics[[0, 3, 6, 9]], images[[0, 3, 6, 9]], v_rgb, 
                weights=weights[[0, 3, 6, 9]], 
                confidence=0.8,
                use_alpha=False,
                use_inpainting=True,
                visualize=visualize,
            )
            map_Kd = None   
    elif not use_vertex and not use_optimize:
        intrinsics = dataset_dict['intrinsic']
        c2ws = dataset_dict['frames_c2w']
        images = dataset_dict['frames_img']

        # NOTE: hard code, elevation=0; azimuth=0,30,60,...,330
        weights = np.array([
            2.0, 0.05, 0.05, 0.5, 0.05, 0.05,
            1.0, 0.05, 0.05, 0.5, 0.05, 0.05,
        ])
        # stage 1: initialization
        if initialize_from_vextex:
            v_rgb = remapping_vertex_color(
                mesh, c2ws, intrinsics, images, v_rgb, 
                weights=[1.0] * len(weights), 
                confidence=0.2,
                use_alpha=False,
                use_inpainting=True,
                visualize=visualize,
            )
            map_Kd = initial_map_Kd_with_v_rgb(
                mesh, v_rgb, 
                texture_size=2048,
                visualize=visualize,
            )
        else:
            map_Kd = remapping_uv_texture(
                mesh, c2ws, intrinsics, images, map_Kd, 
                weights=[1.0] * len(weights), 
                use_alpha=False,
                use_inpainting=True, use_dilation=False,
                texture_size=2048, confidence=0.2,
                visualize=visualize,
            )
        if visualize:
            cv2.imwrite(f'./.cache/map_Kd_result_stage_1.png', map_Kd.mul(255.0)[..., [2,1,0,3]].detach().cpu().numpy())
        # stage 2: refine from 4 views
        map_Kd = remapping_uv_texture(
            mesh, c2ws[[0, 3, 6, 9]], intrinsics[[0, 3, 6, 9]], images[[0, 3, 6, 9]], map_Kd, 
            weights=weights[[0, 3, 6, 9]], 
            use_alpha=False,
            use_inpainting=False, use_dilation=True,
            texture_size=2048, confidence=0.8,
            visualize=visualize,
        )
        if visualize:
            cv2.imwrite(f'./.cache/map_Kd_result_stage_2.png', map_Kd.mul(255.0)[..., [2,1,0,3]].detach().cpu().numpy())
        v_rgb = None

    # export mesh
    if sr_func is not None:
        suffix = '-sr' + suffix
    if use_vertex:
        suffix = '-vert' + suffix
    else:
        suffix = '-uv' + suffix
    if use_optimize:
        suffix = '-opt' + suffix
    else:
        suffix = '-re' + suffix
    export_mesh_path = f'{output_path}/export{suffix}/model.obj'
    texture = Texture(mesh, v_rgb=v_rgb, map_Kd=map_Kd)
    texture.export(export_mesh_path)

    # render video
    export_video_path = f'{output_path}/export{suffix}/video.mp4'
    video_exportter = VideoExporter()
    video_exportter.export_arbit_video(export_mesh_path, export_video_path)
   



def sr_function(images: np.ndarray) -> np.ndarray:
    '''
    images: [N, H, W, C], uint8

    * install hat: 
        cd extern/HAT
        python setup.py develop
        cd ../..
    * link pretrain models:
        ln -s PATH_TO_HAT_PRETRAIN_MODELS/Real_HAT_GAN_sharper.pth \
            extern/HAT/experiments/pretrained_models/Real_HAT_GAN_sharper.pth
    '''
    N, H, W, C = images.shape
    
    # prepare paths
    hat_home = os.path.join(os.path.dirname(__file__), 'extern/HAT')
    hat_config_path = os.path.join(hat_home, "options/test/HAT_GAN_Real_SRx4_figgen.yml")
    hat_pretrain_model_path = os.path.join(hat_home, "experiments/pretrained_models/Real_HAT_GAN_sharper.pth")
    hat_script_path = os.path.join(hat_home, 'hat/test.py')
    root_dir = f"/tmp/figgen/sr"
    config_path = os.path.join(root_dir, "config.yml")
    inputs_dir = os.path.join(root_dir, "inputs")
    results_dir = os.path.join(root_dir, "results")
    if os.path.exists(root_dir):
        shutil.rmtree(root_dir)
    os.makedirs(root_dir, exist_ok=True)

    # dump config to /tmp/figgen/sr/config.yml
    with open(config_path, "w") as f:
        args: OmegaConf = OmegaConf.load(hat_config_path)
        OmegaConf.resolve(args)
        template = args['datasets']['test_1']
        
        datasets_dict = dict()
        for i in range(N):
            template_cur = deepcopy(template)

            dst_dir = f'{inputs_dir}/{i:04d}'
            os.makedirs(dst_dir, exist_ok=True)
            template_cur['dataroot_lq'] = dst_dir
            Image.fromarray(images[i]).save(os.path.join(dst_dir, f'{i:04d}.png'))

            datasets_dict[f'test_{i}'] = template_cur
        args['datasets'] = datasets_dict
        args['path']['pretrain_network_g'] = hat_pretrain_model_path
        args['path']['results_root'] = results_dir
        args['path']['log'] = os.path.join(results_dir, 'log')
        args['path']['visualization'] = os.path.join(results_dir, 'vis')
        f.write(OmegaConf.to_yaml(args))

    # run hat
    cmd = f"python {hat_script_path} -opt {config_path}"
    os.system(cmd)
    
    # load images from /tmp/figgen/sr/results/vis/custom/0000_{args.name}.png
    results_path_list = sorted(glob(os.path.join(results_dir, 'vis/custom/*.png')))
    image_list = []
    for i, path in enumerate(results_path_list):
        image_list.append(np.array(Image.open(path), dtype=np.uint8))
    images_sr = np.stack(image_list, axis=0)
    return images_sr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mesh_path", required=True, help="path to input mesh")
    parser.add_argument("--image_path", required=True, help="path to image data, saved as image_path/*.png")
    parser.add_argument("--camera_path", required=True, help="path to camera data, saved as camera_path/transforms.json or camera_path/c2w.npy & camera_path/proj.npy")
    parser.add_argument("--output_path", required=True, help="path to output files")
    parser.add_argument("--index_list", default=None, help="list of index, such as [0,1,2,3]")
    parser.add_argument("--svsr", action='store_true', help='whether use single-view super resolution')
    parser.add_argument("--mvsr", action='store_true', help='whether use multiview super resolution')
    parser.add_argument("--uv", action='store_true', help="vertex color or uv texture map, default is vertex color")
    parser.add_argument("--reproject", action='store_true', help="reproject method or optimize method, default is optimize method")
    parser.add_argument("--debug", action='store_true', help="whether use debugging mode")
    parser.add_argument(
        "--gpu",
        default="0",
        help="GPU(s) to be used. 0 means use the 1st available GPU. "
        "1,2 means use the 2nd and 3rd available GPU. "
        "If CUDA_VISIBLE_DEVICES is set before calling `launch.py`, "
        "this argument is ignored and all available GPUs are always used.",
    )
    opt = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "%s"%opt.gpu
    
    opt_warpper(
        opt.mesh_path, opt.image_path, opt.camera_path, opt.output_path, 
        index_list=opt.index_list, 
        sr_func=sr_function if opt.svsr else None, 
        use_vertex=not opt.uv,
        use_optimize=not opt.reproject,
        visualize=opt.debug,
    )