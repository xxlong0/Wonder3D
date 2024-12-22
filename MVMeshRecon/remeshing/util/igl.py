import torch
import igl
import numpy as np

@torch.no_grad()
def igl_flips(
        vertices:np.array, #V,3
        faces:np.array, #F,3
        target_vertices:np.array, #VT,3
        target_faces:np.array, #FT,3
    )->tuple[np.array,np.array]:

    full_vertices = vertices[faces] #F,C=3,3
    face_centers = full_vertices.mean(axis=1) #F,3
    _,ind,points = igl.point_mesh_squared_distance(face_centers,target_vertices,target_faces)
    target_faces = target_faces[ind] #F,3
    corners = target_vertices[target_faces] #F,3,3
    bary = igl.barycentric_coordinates_tri(points,corners[:,0].copy(),corners[:,1].copy(),corners[:,2].copy()) #P,3
    target_normals = igl.per_vertex_normals(target_vertices,target_faces,igl.PER_VERTEX_NORMALS_WEIGHTING_TYPE_AREA)
    corner_normals = target_normals[target_faces] #P,3,3
    ref_normals = (bary[:,:,None] * corner_normals).sum(axis=1) #F,3
    face_normals = igl.per_face_normals(vertices,faces,np.array([0,0,0],dtype=np.float32)) #F,3 not normalized
    flip = np.sum(ref_normals * face_normals, axis=-1)<0 #F
    flipped_area = np.sum(flip * np.linalg.norm(face_normals,axis=-1))
    total_area = np.sum(np.linalg.norm(face_normals,axis=-1))
    ratio = flipped_area / total_area
    return flip, ratio


@torch.no_grad()
def igl_distance(
        vertices:np.array, #V,3
        faces:np.array, #F,3
        target_vertices:np.array, #VT,3
        target_faces:np.array, #FT,3
        ):
        
    dist1_sq,_,_ = igl.point_mesh_squared_distance(vertices,target_vertices,target_faces)
    dist2_sq,_,_ = igl.point_mesh_squared_distance(target_vertices,vertices,faces)
    vertex_distance = np.sqrt(dist1_sq)

    rms_distance = ((dist1_sq.mean()+dist2_sq.mean())/2)**.5
    max_distance = max(dist1_sq.max(),dist2_sq.max())**.5

    return vertex_distance,rms_distance,max_distance