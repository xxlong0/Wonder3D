import torch
from core.remesh import calc_face_normals, prepend_dummies

def make_grid(flip:torch.Tensor,device='cuda'): #HW,2
    H,W = flip.shape
    vertices = torch.zeros(((H+1)*(W+1),3),device=device)
    vertices[:,[1,0]] = torch.cartesian_prod(torch.arange(0,H+1),torch.arange(0,W+1)).float().to(device)
    c = (torch.arange(0,W) + (W+1)*torch.arange(0,H)[:,None]).reshape(-1).to(device) #HW
    faces_right = torch.stack((torch.stack((c,c+1,c+1+W+1),dim=-1),torch.stack((c,c+1+W+1,c+W+1),dim=-1)),dim=1) #HW,2,3
    faces_left = torch.stack((torch.stack((c,c+1,c+W+1),dim=-1),torch.stack((c+1,c+1+W+1,c+W+1),dim=-1)),dim=1) #HW,2,3
    faces = torch.where(flip.flipud().reshape(H*W,1,1),faces_left,faces_right)
    faces = faces.reshape(H*W*2,3)
    return prepend_dummies(vertices,faces)
    
def area(vertices,faces):
    return calc_face_normals(vertices,faces)[1:,2].sum().item()/2

