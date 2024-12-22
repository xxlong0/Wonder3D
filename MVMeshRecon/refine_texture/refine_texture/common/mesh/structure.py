import os
from time import perf_counter
from typing import Dict, Optional
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import trimesh
from trimesh.visual import ColorVisuals, TextureVisuals
from trimesh.visual.texture import unmerge_faces
from trimesh.visual.material import SimpleMaterial, PBRMaterial
import pymeshlab as ml
import open3d as o3d

from .utils import dot
from ..renderers.nvdiffrast.renderer import NVDiffRendererBase


class DeviceMixin:
    attr_list = []
    def to(self, device):
        device = torch.device(device)
        for key in self.attr_list:
            value = getattr(self, key, None)
            if value is not None:
                if isinstance(value, torch.Tensor) or (hasattr(value, 'device') and hasattr(value, 'to')):
                    if value.device != device:
                        setattr(self, key, value.to(device))
        return self


class ExporterMixin:
    def to_trimesh(self) -> trimesh.Trimesh:
        raise NotImplementedError

    def to_open3d(self) -> o3d.geometry.TriangleMesh:
        raise NotImplementedError
    
    def _to_pymeshlab(self) -> Dict:
        raise NotImplementedError

    def to_pymeshlab(self) -> ml.Mesh:
        return ml.Mesh(**self._to_pymeshlab())

    def export(self, obj_path: str, backend='trimesh'):
        assert backend in ['trimesh', 'open3d', 'pymeshlab', 'blender']
        obj_path = os.path.abspath(obj_path)
        os.makedirs(os.path.dirname(obj_path), exist_ok=True)
        if backend == 'trimesh':
            self.to_trimesh().export(obj_path)
        elif backend == 'open3d':
            o3d.io.write_triangle_mesh(
                obj_path, 
                self.to_open3d(), 
                write_ascii=False, 
                compressed=False, 
                write_vertex_normals=True, 
                write_vertex_colors=True, 
                write_triangle_uvs=True, 
                print_progress=False,
            )
        elif backend == 'pymeshlab':
            ms = ml.MeshSet()
            ms.add_mesh(self.to_pymeshlab(), mesh_name='model', set_as_current=True)
            ms.save_current_mesh(
                obj_path,
                save_vertex_color = True,
                save_vertex_coord = True,
                save_vertex_normal = True,
                save_face_color = False, 
                save_wedge_texcoord = True,
                save_wedge_normal = False,
                save_polygonal = False,
            )
            map_Kd = getattr(self, 'map_Kd', None)
            if map_Kd is not None:
                image = Image.fromarray(map_Kd.flip(-3).clamp(0.0, 1.0).mul(255.0).detach().cpu().numpy().astype(np.uint8), mode='RGBA')
                image.save(os.path.join(os.path.dirname(obj_path), 'material_0.png'))
                mtl_path = os.path.join(os.path.dirname(obj_path), os.path.basename(obj_path)+'.mtl')
                if os.path.isfile(mtl_path):
                    with open(mtl_path, 'r+') as f:
                        lines = f.readlines()
                        has_mtl = False
                        for i in range(len(lines)):
                            if lines[i].split(' ')[0] == 'map_Kd':
                                lines[i] = f'map_Kd material_0.png'
                                has_mtl = True
                                break
                        if not has_mtl:
                            lines.append(f'map_Kd material_0.png')
                        f.seek(0, 0)
                        f.writelines(lines)
                else:
                    with open(mtl_path, 'w') as f:
                        f.write(f'newmtl material_0\n')
                        f.write(f'Ka 1.00000000 1.00000000 1.00000000\n')
                        f.write(f'Kd 1.00000000 1.00000000 1.00000000\n')
                        f.write(f'Kd 1.00000000 1.00000000 1.00000000\n')
                        f.write(f'Ns 1.00000000\n')
                        f.write(f'map_Kd material_0.png\n')
                    with open(obj_path, "r+") as f:
                        content = f.read()
                        f.seek(0, 0)
                        f.write(f'mtllib {os.path.basename(obj_path)}.mtl\n')
                        f.write(f'usemtl material_0\n')
                        f.write(content)
        else:
            raise NotImplementedError(f'backend {backend} is not supported yet')


class MeasureMixin:  # disable lazy api
    def bbox(self):  # [2, 3]
        return torch.stack([
            self.v_pos.min(dim=0).values,
            self.v_pos.max(dim=0).values,
        ], dim=0)
    
    def bbox_center(self):  # [3,]
        return self.bbox().mean(dim=0)
    
    def bbox_radius(self):  # [3,]
        bbox = self.bbox()
        return (bbox[1, :] - bbox[0, :]).square().sum().sqrt()
    
    def center(self):  # [3,]
        return self.v_pos.mean(dim=0)
    
    def radius(self):
        d2 = self.v_pos.square().sum(dim=1)
        return d2.max().sqrt()


class TransformMixin(DeviceMixin):
    def __init__(self) -> None:
        super().__init__()
        self._identity = torch.eye(n=4, dtype=torch.float32)
        self._transform = None
        self.attr_list.extend(['_identity', '_transform'])
    
    @property
    def identity(self) -> torch.Tensor:  # read only
        return self._identity
    
    def init_transform(self, transform:Optional[torch.Tensor]=None):
        self._transform = self.identity.clone() if transform is None else transform
        return self
    
    def apply_transform(self, clear_transform=True):
        if self._transform is not None:
            v_pos_homo = torch.cat([self.v_pos, torch.ones_like(self.v_pos[:, [0]])], dim=-1)
            v_pos_homo = torch.matmul(v_pos_homo, self._transform.T.to(v_pos_homo))
            self.v_pos = v_pos_homo[:, :3]
            if clear_transform:
                self._transform = None
        return self
    
    def compose_transform(self, transform:torch.Tensor, after=True):
        if self._transform is None:
            self.init_transform()
        if after:
            self._transform = torch.matmul(transform.to(self._transform), self._transform)
        else:
            self._transform = torch.matmul(self._transform, transform.to(self._transform))
        return self


class CoordinateSystemMixin(MeasureMixin, TransformMixin):
    def flip_x(self):
        transform = self.identity.clone()
        transform[0, 0] = -1
        self.compose_transform(transform)
        return self

    def flip_y(self):
        transform = self.identity.clone()
        transform[1, 1] = -1
        self.compose_transform(transform)
        return self

    def flip_z(self):
        transform = self.identity.clone()
        transform[2, 2] = -1
        self.compose_transform(transform)
        return self

    def swap_xy(self):
        transform = torch.zeros_like(self.identity)
        transform[0, 1] = 1
        transform[1, 0] = 1
        transform[2, 2] = 1
        transform[3, 3] = 1
        self.compose_transform(transform)
        return self

    def swap_yz(self):
        transform = torch.zeros_like(self.identity)
        transform[0, 0] = 1
        transform[1, 2] = 1
        transform[2, 1] = 1
        transform[3, 3] = 1
        self.compose_transform(transform)
        return self

    def swap_zx(self):
        transform = torch.zeros_like(self.identity)
        transform[0, 2] = 1
        transform[1, 1] = 1
        transform[2, 0] = 1
        transform[3, 3] = 1
        self.compose_transform(transform)
        return self
    
    def translate(self, offset:torch.Tensor):
        transform = self.identity.clone()
        transform[:3, 3] = offset
        self.compose_transform(transform)
        return self

    def scale(self, scale:torch.Tensor):
        transform = self.identity.clone()
        transform = transform.diagonal_scatter(scale)
        self.compose_transform(transform)
        return self

    def align_to(self, src:torch.Tensor, dst:Optional[torch.Tensor]=None):
        if dst is None:
            return self.translate(src.neg())
        else:
            return self.translate(dst-src)
    
    def scale_to(self, src:torch.Tensor, dst:Optional[torch.Tensor]=None):
        if dst is None:
            return self.scale(src.reciprocal())
        else:
            return self.scale(dst/src)

    def align_to_bbox_center(self):
        return self.align_to(self.bbox_center())
    
    def align_center(self):
        return self.align_to(self.center())


class Mesh(ExporterMixin, CoordinateSystemMixin):
    def __init__(self, v_pos:torch.Tensor, t_pos_idx:torch.Tensor, **kwargs):
        self.v_pos = v_pos
        self.t_pos_idx = t_pos_idx
        self._v_nrm = None
        self._v_tng = None
        self._v_tex = None
        self._t_tex_idx = None
        self._e_pos_idx = None
        self._t_edge_idx = None
        self._e_triangle_idx = None

        assert v_pos.device == t_pos_idx.device, \
            f'v_pos is on {v_pos.device} but t_pos_idx is on {t_pos_idx.device}'
        self.attr_list.extend([
            'v_pos',
            't_pos_idx',
            '_v_nrm',
            '_v_tng',
            '_v_tex',
            '_t_tex_idx',
            '_e_pos_idx',
            '_t_edge_idx',
            '_e_triangle_idx',
        ])
        CoordinateSystemMixin.__init__(self)
    
    @property
    def device(self):
        return self.v_pos.device
    
    @classmethod
    def from_trimesh(cls, mesh: trimesh.Trimesh):
        mesh = mesh.copy(include_cache=True)
        mesh.merge_vertices(merge_tex=True, merge_norm=True)
        v_pos = torch.as_tensor(mesh.vertices, dtype=torch.float32)
        t_pos_idx = torch.as_tensor(mesh.faces, dtype=torch.int64)
        return Mesh(v_pos=v_pos, t_pos_idx=t_pos_idx)
    
    @classmethod
    def from_open3d(cls, mesh: o3d.geometry.TriangleMesh):
        # TODO: merge_vertices
        if isinstance(mesh, o3d.t.geometry.TriangleMesh):
            mesh = mesh.to_legacy()
        v_pos = torch.as_tensor(np.asarray(mesh.vertices, dtype=np.float32), dtype=torch.float32)
        t_pos_idx = torch.as_tensor(np.asarray(mesh.triangles, dtype=np.int64), dtype=torch.int64)
        return Mesh(v_pos=v_pos, t_pos_idx=t_pos_idx)
    
    def to_trimesh(self) -> trimesh.Trimesh:
        mesh = trimesh.Trimesh(
            vertices=self.v_pos.detach().cpu().numpy(),
            faces=self.t_pos_idx.cpu().numpy(),
            process=False,
        )
        #
        # m = self.merge_faces()
        # if m._v_tex is not None and m._t_tex_idx is not None:
        #     uv = m.v_tex.detach().cpu().numpy()
        # else:
        #     uv = None
        # mesh.visual = TextureVisuals(
        #     uv=uv,
        #     material=PBRMaterial(
        #         baseColorTexture=None,
        #         metallicFactor=0.9,
        #         baseColorFactor=[1., 1., 1., 1.]
        #     ),
        # )
        return mesh
    
    def to_open3d(self) -> o3d.geometry.TriangleMesh:
        mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(self.v_pos.detach().cpu().numpy()), 
            triangles=o3d.utility.Vector3iVector(self.t_pos_idx.cpu().numpy()),
        )
        return mesh
    
    def _to_pymeshlab(self) -> Dict:
        return dict(
            vertex_matrix=self.v_pos.detach().cpu().numpy(),
            face_matrix=self.t_pos_idx.cpu().numpy(),
        )

    def remove_outlier(self, outlier_n_faces_threshold):
        if self.requires_grad:
            print("Mesh is differentiable, not removing outliers")
            return self

        # use trimesh to first split the mesh into connected components
        # then remove the components with less than n_face_threshold faces
        import trimesh

        # construct a trimesh object
        mesh = trimesh.Trimesh(
            vertices=self.v_pos.detach().cpu().numpy(),
            faces=self.t_pos_idx.detach().cpu().numpy(),
        )

        # split the mesh into connected components
        components = mesh.split(only_watertight=False)
        # log the number of faces in each component
        print(
            "Mesh has {} components, with faces: {}".format(
                len(components), [c.faces.shape[0] for c in components]
            )
        )

        n_faces_threshold: int
        if isinstance(outlier_n_faces_threshold, float):
            # set the threshold to the number of faces in the largest component multiplied by outlier_n_faces_threshold
            n_faces_threshold = int(
                max([c.faces.shape[0] for c in components]) * outlier_n_faces_threshold
            )
        else:
            # set the threshold directly to outlier_n_faces_threshold
            n_faces_threshold = outlier_n_faces_threshold

        # log the threshold
        print(
            "Removing components with less than {} faces".format(n_faces_threshold)
        )

        # remove the components with less than n_face_threshold faces
        components = [c for c in components if c.faces.shape[0] >= n_faces_threshold]

        # log the number of faces in each component after removing outliers
        print(
            "Mesh has {} components after removing outliers, with faces: {}".format(
                len(components), [c.faces.shape[0] for c in components]
            )
        )
        # merge the components
        mesh = trimesh.util.concatenate(components)

        # convert back to our mesh format
        v_pos = torch.from_numpy(mesh.vertices).to(self.v_pos)
        t_pos_idx = torch.from_numpy(mesh.faces).to(self.t_pos_idx)

        clean_mesh = Mesh(v_pos, t_pos_idx)
        # keep the extras unchanged

        if len(self.extras) > 0:
            clean_mesh.extras = self.extras
            print(
                f"The following extra attributes are inherited from the original mesh unchanged: {list(self.extras.keys())}"
            )
        return clean_mesh

    def merge_faces(self, v_pos_attr=None, v_tex_attr=None):
        if self._v_tex is not None and self._t_tex_idx is not None:
            t_joint_idx, v_pos_idx, v_tex_idx = unmerge_faces(self.t_pos_idx.cpu().numpy(), self.t_tex_idx.cpu().numpy(), maintain_faces=False)
            t_joint_idx = torch.as_tensor(t_joint_idx, dtype=torch.int64, device=self.device)
            v_pos_idx = torch.as_tensor(v_pos_idx, dtype=torch.int64, device=self.device)
            v_tex_idx = torch.as_tensor(v_tex_idx, dtype=torch.int64, device=self.device)
            m = Mesh(v_pos=self.v_pos[v_pos_idx], t_pos_idx=t_joint_idx)
            m._v_tex = self._v_tex[v_tex_idx]
            m._t_tex_idx = t_joint_idx
            if v_pos_attr is None and v_tex_attr is None:
                return m
            else:
                if v_pos_attr is not None:
                    v_pos_attr = v_pos_attr[v_pos_idx]
                if v_tex_attr is not None:
                    v_tex_attr = v_tex_attr[v_tex_idx]
                return m, v_pos_attr, v_tex_attr
        else:
            if v_pos_attr is None and v_tex_attr is None:
                return self
            else:
                return self, v_pos_attr, v_tex_attr

    @property
    def v_nrm(self) -> torch.Tensor:
        if self._v_nrm is None:
            self._v_nrm = self._compute_vertex_normal()
        return self._v_nrm

    @property
    def v_tng(self) -> torch.Tensor:
        if self._v_tng is None:
            self._v_tng = self._compute_vertex_tangent()
        return self._v_tng

    @property
    def v_tex(self) -> torch.Tensor:
        if self._v_tex is None:
            self.unwrap_uv()
        return self._v_tex

    @property
    def t_tex_idx(self) -> torch.Tensor:
        if self._t_tex_idx is None:
            self.unwrap_uv()
        return self._t_tex_idx

    @property
    def e_pos_idx(self) -> torch.Tensor:
        if self._e_pos_idx is None:
            self._compute_edges()
        return self._e_pos_idx
    
    @property
    def t_edge_idx(self) -> torch.Tensor:
        if self._t_edge_idx is None:
            self._compute_edges()
        return self._t_edge_idx

    @property
    def e_triangle_idx(self) -> torch.Tensor:
        raise NotImplementedError
        if self._e_triangle_idx is None:
            self._compute_edges()
        return self._e_triangle_idx

    def _compute_vertex_normal(self):
        i0 = self.t_pos_idx[:, 0].long()
        i1 = self.t_pos_idx[:, 1].long()
        i2 = self.t_pos_idx[:, 2].long()

        v0 = self.v_pos[i0, :]
        v1 = self.v_pos[i1, :]
        v2 = self.v_pos[i2, :]

        face_normals = torch.linalg.cross(v1 - v0, v2 - v0)

        # Splat face normals to vertices
        v_nrm = torch.zeros_like(self.v_pos)
        v_nrm.scatter_add_(0, i0[:, None].repeat(1, 3), face_normals)
        v_nrm.scatter_add_(0, i1[:, None].repeat(1, 3), face_normals)
        v_nrm.scatter_add_(0, i2[:, None].repeat(1, 3), face_normals)

        # Normalize, replace zero (degenerated) normals with some default value
        v_nrm = torch.where(
            dot(v_nrm, v_nrm) > 1e-20, v_nrm, torch.as_tensor([0.0, 0.0, 1.0]).to(v_nrm)
        )
        v_nrm = F.normalize(v_nrm, dim=1)

        if torch.is_anomaly_enabled():
            assert torch.all(torch.isfinite(v_nrm))

        return v_nrm

    def _compute_vertex_tangent(self):
        vn_idx = [None] * 3
        pos = [None] * 3
        tex = [None] * 3
        for i in range(0, 3):
            pos[i] = self.v_pos[self.t_pos_idx[:, i]]
            tex[i] = self.v_tex[self.t_tex_idx[:, i]]
            # t_nrm_idx is always the same as t_pos_idx
            vn_idx[i] = self.t_pos_idx[:, i]

        tangents = torch.zeros_like(self.v_nrm)
        tansum = torch.zeros_like(self.v_nrm)

        # Compute tangent space for each triangle
        uve1 = tex[1] - tex[0]
        uve2 = tex[2] - tex[0]
        pe1 = pos[1] - pos[0]
        pe2 = pos[2] - pos[0]

        nom = pe1 * uve2[..., 1:2] - pe2 * uve1[..., 1:2]
        denom = uve1[..., 0:1] * uve2[..., 1:2] - uve1[..., 1:2] * uve2[..., 0:1]

        # Avoid division by zero for degenerated texture coordinates
        tang = nom / torch.where(
            denom > 0.0, torch.clamp(denom, min=1e-6), torch.clamp(denom, max=-1e-6)
        )

        # Update all 3 vertices
        for i in range(0, 3):
            idx = vn_idx[i][:, None].repeat(1, 3)
            tangents.scatter_add_(0, idx, tang)  # tangents[n_i] = tangents[n_i] + tang
            tansum.scatter_add_(
                0, idx, torch.ones_like(tang)
            )  # tansum[n_i] = tansum[n_i] + 1
        tangents = tangents / tansum

        # Normalize and make sure tangent is perpendicular to normal
        tangents = F.normalize(tangents, dim=1)
        tangents = F.normalize(tangents - dot(tangents, self.v_nrm) * self.v_nrm)

        if torch.is_anomaly_enabled():
            assert torch.all(torch.isfinite(tangents))

        return tangents

    def _unwrap_uv_v1(self, chart_config:Dict=dict(), pack_config:Dict=dict()):
        import xatlas

        atlas = xatlas.Atlas()
        v_pos = self.laplacian_func(self.v_pos, depth=3)
        atlas.add_mesh(
            v_pos.detach().cpu().numpy(),
            self.t_pos_idx.cpu().numpy(),
        )
        _chart_config = {
            'max_chart_area': 0.0,
            'max_boundary_length': 0.0,
            'normal_deviation_weight': 2.0,
            'roundness_weight': 0.01,
            'straightness_weight': 6.0,
            'normal_seam_weight': 4.0,
            'texture_seam_weight': 0.5,
            'max_cost': 16.0,  # avoid small charts
            'max_iterations': 1,
            'use_input_mesh_uvs': False,
            'fix_winding': False,
        }
        _pack_config = {
            'max_chart_size': 0,
            'padding': 4,  # avoid adjoint
            'texels_per_unit': 0.0,
            'resolution': 2048,
            'bilinear': True,
            'blockAlign': False,
            'bruteForce': False,
            'create_image': False,
            'rotate_charts_to_axis': True,
            'rotate_charts': True,
        }
        _chart_config.update(chart_config)
        _pack_config.update(pack_config)
        co = xatlas.ChartOptions()
        po = xatlas.PackOptions()
        for k, v in _chart_config.items():
            setattr(co, k, v)
        for k, v in _pack_config.items():
            setattr(po, k, v)
        
        print(f"UV unwrapping, V={self.v_pos.shape[0]}, F={self.t_pos_idx.shape[0]}, may take a while ...")
        t = perf_counter()
        atlas.generate(co, po, verbose=False)
        print(f"UV unwrapping wastes {perf_counter() - t} sec")

        _, indices, uvs = atlas.get_mesh(0)
        uvs = torch.as_tensor(uvs.astype(np.float32), dtype=self.v_pos.dtype, device=self.v_pos.device)
        indices = torch.as_tensor(indices.astype(np.int64), dtype=self.t_pos_idx.dtype, device=self.t_pos_idx.device)
        return uvs, indices
    
    def _unwrap_uv_v2(self, config:Dict=dict()):
        device = o3d.core.Device('CPU:0')
        dtype_f = o3d.core.float32
        dtype_i = o3d.core.int64
        mesh = o3d.t.geometry.TriangleMesh(device=device)
        v_pos = self.laplacian_func(self.v_pos, depth=3)
        mesh.vertex.positions = o3d.core.Tensor(v_pos.detach().cpu().numpy(), dtype=dtype_f, device=device)
        mesh.triangle.indices = o3d.core.Tensor(self.t_pos_idx.cpu().numpy(), dtype=dtype_i, device=device)
        _config = {
            'size': 2048,
            'gutter': 4.0,
            'max_stretch': 0.1667,
            'parallel_partitions': 4,
            'nthreads': 0,
        }
        _config.update(config)
        
        print(f"UV unwrapping, V={self.v_pos.shape[0]}, F={self.t_pos_idx.shape[0]}, may take a while ...")
        t = perf_counter()
        mesh.compute_uvatlas(**_config)
        print(f"UV unwrapping wastes {perf_counter() - t} sec")

        triangle_uvs = mesh.triangle.texture_uvs.numpy().astype(np.float32)  # [F*3, 2]
        triangle_uvs = triangle_uvs.reshape(self.t_pos_idx.shape[0], 3, 2)
        t_tex = torch.as_tensor(triangle_uvs, dtype=self.v_pos.dtype, device=self.v_pos.device)  # [F, 3, 2]
        v_tex_full = t_tex.reshape(self.t_pos_idx.shape[0] * 3, 2)
        t_tex_idx_full = torch.arange(self.t_pos_idx.shape[0] * 3, dtype=self.t_pos_idx.dtype, device=self.t_pos_idx.device).reshape(self.t_pos_idx.shape[0], 3)
        v_tex, v_tex_idx = torch.unique(v_tex_full, dim=0, sorted=False, return_inverse=True, return_counts=False)
        t_tex_idx = v_tex_idx[t_tex_idx_full]
        return v_tex, t_tex_idx
        
    def unwrap_uv(self, **kwargs):
        self._v_tex, self._t_tex_idx = self._unwrap_uv_v2(**kwargs)

    def _compute_edges(self):
        e_pos_idx_full = torch.cat([self.t_pos_idx[:, [0, 1]], self.t_pos_idx[:, [1, 2]], self.t_pos_idx[:, [2, 0]]], dim=0)  # [F*3, 2]
        e_pos_idx_sorted, _ = torch.sort(e_pos_idx_full, dim=-1)  # [F*3, 2], [F*3, 2]
        e_pos_idx, _t_edge_idx = torch.unique(e_pos_idx_sorted, dim=0, sorted=False, return_inverse=True, return_counts=False)  # [E, 2], [F*3,]
        t_edge_idx = _t_edge_idx.reshape(self.t_pos_idx.shape[0], 3)    
        # e_triangle_idx = torch.zeros((e_pos_idx.shape[0], 2), dtype=self.t_pos_idx.dtype, device=self.t_pos_idx.device)  # [E, 2]
        # dart_index = torch.cartesian_prod(
        #     torch.arange(F, dtype=self.t_pos_idx.dtype, device=self.t_pos_idx.device), 
        #     torch.arange(3, dtype=self.t_pos_idx.dtype, device=self.t_pos_idx.device), 
        # )  # [F*3, 2]
        # e_triangle_idx = e_triangle_idx.scatter(0, _t_edge_idx.unsqueeze(-1), dart_index)  # e_triangle_idx[_t_edge_idx[ii, j], j] = dart_index[ii, j]
        self._e_pos_idx = e_pos_idx
        self._t_edge_idx = t_edge_idx
        self._e_triangle_idx = None  # FIXME

    def normal_consistency(self):
        edge_nrm = self.v_nrm[self.e_pos_idx]  # [E, 2, 3]
        nc = (1.0 - torch.cosine_similarity(edge_nrm[:, 0, :], edge_nrm[:, 1, :], dim=-1)).mean()
        return nc

    def _laplacian_v1(self, reciprocal=False):
        verts, faces = self.v_pos, self.t_pos_idx
        V = verts.shape[0]
        F = faces.shape[0]

        # neighbor
        ii = faces[:, [1, 2, 0]].flatten()
        jj = faces[:, [2, 0, 1]].flatten()
        adj_idx = torch.stack([torch.cat([ii, jj]), torch.cat([jj, ii])], dim=0).unique(dim=1)
        adj_val = -torch.ones(adj_idx.shape[1]).to(verts)

        # diagonal
        diag_idx = torch.stack((adj_idx[0], adj_idx[0]), dim=0)
        diag_val = torch.ones(adj_idx.shape[1]).to(verts)

        # sparse matrix
        idx = torch.cat([adj_idx, diag_idx], dim=1)
        val = torch.cat([adj_val, diag_val], dim=0)
        L = torch.sparse_coo_tensor(idx, val, (V, V))

        # coalesce operation sums the duplicate indices
        L = L.coalesce()
        return L

    def _laplacian_v2(self, reciprocal=False):
        V = self.v_pos.shape[0]
        e0, e1 = self.e_pos_idx.unbind(1)

        idx01 = torch.stack([e0, e1], dim=1)  # (E, 2)
        idx10 = torch.stack([e1, e0], dim=1)  # (E, 2)
        idx = torch.cat([idx01, idx10], dim=0).t()  # (2, 2*E)

        ones = torch.ones(idx.shape[1], dtype=torch.float32, device=self.device)
        A = torch.sparse_coo_tensor(idx, ones, (V, V), dtype=torch.float32, device=self.device)
        deg = torch.sparse.sum(A, dim=1).to_dense()
        if reciprocal:
            deg = torch.nan_to_num(torch.reciprocal(deg), nan=0.0, posinf=0.0, neginf=0.0)
        val = torch.cat([deg[e0], deg[e1]])
        L = torch.sparse_coo_tensor(idx, val, (V, V), dtype=torch.float32, device=self.device)

        # idx = torch.arange(V, device=self.device)
        # idx = torch.stack([idx, idx], dim=0)
        # ones = torch.ones(idx.shape[1], dtype=torch.float32, device=self.device)
        # L -= torch.sparse_coo_tensor(idx, ones, (V, V), dtype=torch.float32, device=self.device)
        return L
    
    def _laplacian_v3(self, reciprocal=False):
        ...  # TODO: torch_scatter
    
    @torch.no_grad()
    def laplacian(self, reciprocal=False):
        '''
        L: [V, V], edge laplacian
            L[i, j] = sum_{k\in V} 1_{k \in N_V(i, 1)}, (i, j) is a edge
            L[i, j] = 0, i == j
            L[i, j] = 0, otherwise
        if reciprocal is True, return 1 / L
        '''
        return self._laplacian_v2(reciprocal=reciprocal)

    def laplacian_func(self, v_attr:torch.Tensor, depth=1):
        if depth == 1:
            return v_attr
        L = self.laplacian(reciprocal=True)
        v_attr = torch.matmul(L, v_attr)
        return self.laplacian_func(v_attr, depth=depth-1)

    def laplacian_loss(self, v_attr:torch.Tensor, depth=1):
        return self.laplacian_func(v_attr, depth=depth).norm(dim=-1).mean()

    def compute_uv_mask(self, render_size:int):
        nvdiffrast_renderer = NVDiffRendererBase()
        _render_size = min(render_size, 2048)
        out = nvdiffrast_renderer.simple_inverse_rendering(
            self, None, None,
            None, None, _render_size,
            enable_antialis=False,
        )
        uv_mask = out['mask']
        if _render_size < render_size:
            uv_mask = torch.nn.functional.interpolate(
                uv_mask.to(dtype=torch.float32).permute(0, 3, 1, 2), 
                size=(render_size, render_size),
                mode='nearest',
            ).to(dtype=torch.bool)
        return uv_mask[0].to(device=self.device)


class Texture(DeviceMixin, ExporterMixin):
    texture_key_dict = {
        'map_kd': 'map_Kd',
        'map_pm': 'map_Pm',
        'map_pr': 'map_Pr',
        'map_ns': 'map_Ns',
        'map_refl': 'map_refl',
    }
    texture_suffix_dict = {
        'map_kd': 'diffuse',
        'map_pm': 'metallic',
        'map_pr': 'roughness',
        'map_ns': 'roughness',
        'map_refl': 'metallic',
    }
    def __init__(
        self, 
        mesh:Mesh, 
        v_rgb:Optional[torch.Tensor]=None, 
        map_Kd:Optional[torch.Tensor]=None,
        map_Ks:Optional[torch.Tensor]=None,
        **kwargs,
    ) -> None:
        self.mesh: Mesh = mesh
        self.v_rgb = v_rgb
        self.map_Kd = map_Kd
        self.map_Ks = map_Ks

        assert v_rgb is None or mesh.device == v_rgb.device, \
            f'mesh is on {mesh.device} but v_rgb is on {v_rgb.device}'
        assert map_Kd is None or mesh.device == map_Kd.device, \
            f'mesh is on {mesh.device} but map_Kd is on {map_Kd.device}'
        assert map_Ks is None or mesh.device == map_Ks.device, \
            f'mesh is on {mesh.device} but map_Ks is on {map_Ks.device}'
        self.attr_list.extend(['mesh', 'v_rgb', 'map_Kd', 'map_Ks'])
    
    @property
    def device(self):
        return self.mesh.device

    def to_trimesh(self) -> trimesh.Trimesh:
        if self.v_rgb is not None:
            mesh:trimesh.Trimesh = self.mesh.to_trimesh()
            mesh.visual = ColorVisuals(
                vertex_colors=self.v_rgb.clamp(0.0, 1.0).detach().cpu().numpy(),
            )
        elif (self.mesh._v_tex is not None and self.mesh._t_tex_idx is not None) or self.map_Kd is not None:
            m = self.mesh.merge_faces()
            mesh:trimesh.Trimesh = m.to_trimesh()
            if m._v_tex is not None and m._t_tex_idx is not None:
                uv = m.v_tex.detach().cpu().numpy()
            else:
                uv = None
            if self.map_Kd is not None:
                image = Image.fromarray(self.map_Kd.flip(-3).clamp(0.0, 1.0).mul(255.0).detach().cpu().numpy().astype(np.uint8), mode='RGBA')
            else:
                image = None
            mesh.visual = TextureVisuals(
                uv=uv,
                # material=SimpleMaterial(
                #     image=image,
                # ),
                material=PBRMaterial(
                    baseColorTexture=image,
                    metallicFactor=0.9,
                    baseColorFactor=[1.,1.,1.,1.]
                ),
            )
        return mesh
    
    def to_open3d(self) -> o3d.geometry.TriangleMesh:
        mesh:o3d.geometry.TriangleMesh = self.mesh.to_open3d()
        if self.v_rgb is not None:
            mesh.vertex_colors = o3d.utility.Vector3dVector(self.v_rgb[..., :3].clamp(0.0, 1.0).detach().cpu().numpy())
        if self.mesh._v_tex is not None and self.mesh._t_tex_idx is not None:
            f_v_uv = self.mesh.v_tex[self.mesh.t_tex_idx]
            mesh.triangle_uvs = o3d.utility.Vector2dVector(f_v_uv.reshape(-1, 2).detach().cpu().numpy())
        if self.map_Kd is not None:
            image = Image.fromarray(self.map_Kd.clamp(0.0, 1.0).mul(255.0).detach().cpu().numpy().astype(np.uint8), mode='RGBA')
            mesh.textures = [o3d.geometry.Image(np.array(image, dtype=np.uint8))]
        return mesh
    
    def _to_pymeshlab(self) -> Dict:
        if self.v_rgb is not None:
            m, v_rgb, _ = self.mesh.merge_faces(self.v_rgb)
        else:
            m = self.mesh.merge_faces()
            v_rgb = None
        mesh: Dict = m._to_pymeshlab()
        if v_rgb is not None:
            mesh['v_color_matrix'] = v_rgb[..., :3].clamp(0.0, 1.0).detach().cpu().numpy()
        if m._v_tex is not None and m._t_tex_idx is not None:
            mesh['v_tex_coords_matrix'] = m.v_tex.detach().cpu().numpy()
        return mesh
    
    @classmethod
    def from_trimesh(cls, mesh: trimesh.Trimesh):
        mesh = mesh.copy(include_cache=True)
        mesh.merge_vertices(merge_tex=False, merge_norm=True)
        if isinstance(mesh.visual, ColorVisuals):
            if mesh.visual.vertex_colors is not None:
                v_rgb = torch.as_tensor(mesh.visual.vertex_colors, dtype=torch.float32)
                if v_rgb.shape[-1] == 1:
                    v_rgb = v_rgb.tile(1, 3)
                elif v_rgb.shape[-1] == 2:
                    v_rgb = torch.cat([
                        v_rgb[..., [0]], 
                        torch.zeros_like(v_rgb[..., [0]]), 
                        v_rgb[..., [1]],
                    ], dim=-1)
                elif v_rgb.shape[-1] == 3:
                    v_rgb = v_rgb
                elif v_rgb.shape[-1] > 3:
                    v_rgb = v_rgb[..., :3]
                else:
                    raise NotImplementedError
            else:
                v_rgb = None
            map_Kd = None
            uv = None
            faces = None
        elif isinstance(mesh.visual, TextureVisuals):
            v_rgb = None
            if isinstance(mesh.visual.material, SimpleMaterial):
                map_Kd = mesh.visual.material.image
            elif isinstance(mesh.visual.material, PBRMaterial):
                map_Kd = mesh.visual.material.baseColorTexture
            else:
                map_Kd = None
            if map_Kd is not None:
                map_Kd = torch.as_tensor(np.array(map_Kd.convert('RGBA'), dtype=np.float32), dtype=torch.float32).div(255.0).flip(-3)
            if mesh.visual.uv is not None:
                uv = torch.as_tensor(mesh.visual.uv, dtype=torch.float32)
                faces = torch.as_tensor(mesh.faces, dtype=torch.int64)
            else:
                uv = None
                faces = None
        else:
            v_rgb = None
            map_Kd = None
            uv = None
            faces = None
        mesh = Mesh.from_trimesh(mesh)
        if uv is not None:
            mesh._v_tex = uv
            mesh._t_tex_idx = faces
        return Texture(mesh=mesh, v_rgb=v_rgb, map_Kd=map_Kd, map_Ks=None)

    @classmethod
    def from_open3d(cls, mesh: o3d.geometry.TriangleMesh):
        ...  # TODO
    
    @classmethod
    def from_pymeshlab(cls, mesh: ml.Mesh):
        ...  # TODO
    
    def reset_map_Kd_mask(self):
        H, W, _ = self.map_Kd.shape
        self.map_Kd[:, :, [-1]] = self.mesh.compute_uv_mask(H).to(self.map_Kd)
