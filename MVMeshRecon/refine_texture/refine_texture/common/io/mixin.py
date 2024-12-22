import os
from typing import List
import cv2
import numpy as np
from ..mesh.structure import Mesh, Texture

class MeshSaver:
    def __init__(self, texture:Texture, pbr_model=None, with_uv=False):
        self.texture = texture
        self.pbr_model = pbr_model
        self.with_uv = with_uv
    
    def export(self, path):
        save_mesh(self.texture, path, with_uv=self.with_uv)
        if self.pbr_model is not None:
            self.pbr_model.export(os.path.dirname(path))


def save_mesh(texture:Texture, path, with_uv=False):
    if not with_uv:  # force exporting ply format for mesh without uv
        mesh = texture.to_trimesh()
        mesh.export(os.path.splitext(path)[0] + ".ply", "ply")
    else:
        obj_saver = ObjSaver(os.path.join(os.path.dirname(path), os.path.splitext(os.path.basename(path))[0]))
        with_material = texture.map_Kd is not None or texture.map_Ks is not None
        if not with_material:
            obj_saver.save_obj(
                filename='trimesh.obj',
                v_pos = texture.mesh.v_pos.detach().cpu().numpy(),
                t_pos_idx = texture.mesh.t_pos_idx.cpu().numpy(),
                v_tex = texture.mesh.v_tex.detach().cpu().numpy(),
                t_tex_idx = texture.mesh.t_tex_idx.cpu().numpy(),
                matname = None, 
                mtllib = None,
            )
        else:
            obj_saver.save_obj(
                filename='trimesh.obj',
                v_pos = texture.mesh.v_pos.detach().cpu().numpy(),
                t_pos_idx = texture.mesh.t_pos_idx.cpu().numpy(),
                v_tex = texture.mesh.v_tex.detach().cpu().numpy(),
                t_tex_idx = texture.mesh.t_tex_idx.cpu().numpy(),
                matname = 'material_0', 
                mtllib = 'material.mtl',
            )
            obj_saver.save_mtl(
                filename ='material.mtl',
                matname ='material_0', 
                Ka = (1.0, 1.0, 1.0), 
                Kd = (0.8, 0.8, 0.8), 
                Ks = (1.0, 1.0, 1.0), 
                map_Kd = texture.map_Kd[:, :, [2,1,0]].cpu().numpy() * 255 if texture.map_Kd is not None else None, 
                map_Ks = texture.map_Ks[:, :, [2,1,0]].cpu().numpy() * 255 if texture.map_Ks is not None else None, 
                map_Bump = None, 
                map_Pm = None, 
                map_Pr = None, 
                map_format = "png",
            )


class ObjSaver():
    def __init__(self, save_dir) -> None:
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir

    def save_obj(
        self,
        filename,
        v_pos,
        t_pos_idx,
        v_nrm=None,
        v_tex=None,
        t_tex_idx=None,
        v_rgb=None,
        matname=None,
        mtllib=None,
    ) -> str:
        obj_save_path = os.path.join(self.save_dir, filename)

        obj_str = ""
        if matname is not None:
            obj_str += f"mtllib {mtllib}\n"
            obj_str += f"g object\n"
            obj_str += f"usemtl {matname}\n"
        for i in range(len(v_pos)):
            obj_str += f"v {v_pos[i][0]} {v_pos[i][1]} {v_pos[i][2]}"
            if v_rgb is not None:
                obj_str += f" {v_rgb[i][0]} {v_rgb[i][1]} {v_rgb[i][2]}"
            obj_str += "\n"
        if v_nrm is not None:
            for v in v_nrm:
                obj_str += f"vn {v[0]} {v[1]} {v[2]}\n"
        if v_tex is not None:
            for v in v_tex:
                obj_str += f"vt {v[0]} {1.0 - v[1]}\n"

        for i in range(len(t_pos_idx)):
            obj_str += "f"
            for j in range(3):
                obj_str += f" {t_pos_idx[i][j] + 1}/"
                if v_tex is not None:
                    obj_str += f"{t_tex_idx[i][j] + 1}"
                obj_str += "/"
                if v_nrm is not None:
                    obj_str += f"{t_pos_idx[i][j] + 1}"
            obj_str += "\n"

        with open(obj_save_path, "w") as f:
            f.write(obj_str)
        return obj_save_path

    def save_mtl(
        self,
        filename,
        matname,
        Ka=(1.0, 1.0, 1.0),
        Kd=(0.8, 0.8, 0.8),
        Ks=(1.0, 1.0, 1.0),
        map_Kd=None,
        map_Ks=None,
        map_Bump=None,
        map_Pm=None,
        map_Pr=None,
        map_format="png",
    ) -> List[str]:
        mtl_save_path = os.path.join(self.save_dir, filename)
        save_paths = [mtl_save_path]

        mtl_str = f"newmtl {matname}\n"
        mtl_str += f"Ka {Ka[0]} {Ka[1]} {Ka[2]}\n"
        if map_Kd is not None:
            map_Kd_save_path = os.path.join(
                os.path.dirname(mtl_save_path), f"{matname}_kd.{map_format}"
            )
            mtl_str += f"map_Kd {matname}_kd.{map_format}\n"
            cv2.imwrite(map_Kd_save_path, map_Kd)
            save_paths.append(map_Kd_save_path)
        else:
            mtl_str += f"Kd {Kd[0]} {Kd[1]} {Kd[2]}\n"
        
        if map_Ks is not None:
            map_Ks_save_path = os.path.join(
                os.path.dirname(mtl_save_path), f"{matname}_ks.{map_format}"
            )
            mtl_str += f"map_Ks {matname}_ks.{map_format}\n"
            cv2.imwrite(map_Ks_save_path, map_Ks)
            save_paths.append(map_Ks_save_path)
        else:
            mtl_str += f"Ks {Ks[0]} {Ks[1]} {Ks[2]}\n"
        
        if map_Bump is not None:
            map_Bump_save_path = os.path.join(
                os.path.dirname(mtl_save_path), f"{matname}_nrm.{map_format}"
            )
            mtl_str += f"map_Bump {matname}_nrm.{map_format}\n"
            cv2.imwrite(map_Bump_save_path, map_Bump)
            save_paths.append(map_Bump_save_path)
        
        if map_Pm is not None:
            map_Pm_save_path = os.path.join(
                os.path.dirname(mtl_save_path), f"{matname}_metallic.{map_format}"
            )
            mtl_str += f"map_Pm {matname}_metallic.{map_format}\n"
            cv2.imwrite(map_Pm_save_path, map_Pm)
            save_paths.append(map_Pm_save_path)
        
        if map_Pr is not None:
            map_Pr_save_path = os.path.join(
                os.path.dirname(mtl_save_path), f"{matname}_roughness.{map_format}"
            )
            mtl_str += f"map_Pr {matname}_roughness.{map_format}\n"
            cv2.imwrite(map_Pr_save_path, map_Pr)
            save_paths.append(map_Pr_save_path)
        
        with open(mtl_save_path, "w") as f:
            f.write(mtl_str)
        return save_paths