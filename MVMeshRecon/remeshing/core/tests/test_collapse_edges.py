import unittest
import torch
from torch import nan
from core.remesh import calc_face_normals, collapse_edges, calc_edges
from core.tests.grid import area, make_grid
import matplotlib.pyplot as plt
import os

#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

device='cuda'

def tensor(*args, **kwargs):
    return torch.tensor(*args, device=device, **kwargs)
    
def plot(src_vertices:torch.Tensor,src_faces:torch.Tensor,vertices:torch.Tensor=None,faces:torch.Tensor=None):
    def p(vertices,faces,shrink,color):
        f = vertices[faces] #F,3,3
        f = torch.lerp(f, f.mean(dim=1,keepdim=True), shrink)
        f = f.cpu().numpy()
        plt.fill(f[:,:,0].T,f[:,:,1].T, edgecolor=color, fill=False)
    p(src_vertices,src_faces,.05,'b')
    if vertices is not None:
        p(vertices,faces,.1,'r')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

class TestCollapseEdgesSafe(unittest.TestCase):

    def test_collapse_single(self):
        #  +y
        #
        #  3-----4      3         
        #  | \   |     / \        
        #  |   \ |    /   \
        #  1-----2   1-----2      +x
        #
        vertices = tensor([
                [nan,nan,nan], #dummy
                [0.,0,0],
                [1,0,0],
                [0,1,0],
                [1,1,0],
            ])   

        faces = tensor([
                [0,0,0], #dummy
                [1,2,3],
                [0,0,0], #unused
                [2,4,3],
            ]) 

        edges,_ = calc_edges(faces)

        collapses = torch.zeros(edges.shape[0],dtype=torch.bool,device=device)
        collapses[5] = True 
        self.assertTrue(edges[5].equal(tensor([3,4])))
        
        vertices_expected = tensor([
                [nan,nan,nan], #dummy
                [0.,0,0],
                [1,0,0],
                [.5,1,0],
                [0,0,0],
            ])

        faces_expected = tensor([
                [0,0,0], #unused
                [1,2,3],
                [0,0,0], #unused
                [0,0,0], #collapsed
            ])

        vertices,faces = collapse_edges(vertices,faces,edges,collapses.float(),stable=True)

        self.assertTrue(vertices[:4].allclose(vertices_expected[:4],equal_nan=True))
        self.assertTrue(faces.equal(faces_expected))

    def test_collapse_connectivity(self):        
        flip = torch.zeros((2,3),dtype=torch.bool,device=device)
        vertices,faces = make_grid(flip)
        faces[(faces-1)%4==3] -= 3
        faces[0] = 0
        #plot(vertices,faces)
        edges,_ = calc_edges(faces)
        src_vertices,src_faces = vertices.clone(),faces.clone()
        priorities = torch.zeros(edges.shape[0],device=device)
        priorities[10] = 1
        self.assertTrue((vertices[edges[10],1]==1).all().item())
        vertices,faces = collapse_edges(vertices,faces,edges,priorities)
        self.assertTrue(vertices.allclose(src_vertices,equal_nan=True))
        self.assertTrue(faces.equal(src_faces))

    def test_collapse_random(self):        
        for _ in range(5):
            for w in range(1,20):
                flip = torch.randint(0,2,(w,w),dtype=torch.bool,device=device)
                vertices,faces = make_grid(flip)
                edges,_ = calc_edges(faces)

                self.assertEqual(area(vertices,faces),w*w)
                src_vertices,src_faces = vertices.clone(),faces.clone()
                priorities = torch.rand(edges.shape[0],device=device)-.5
                priorities[0] = 0
                vertices,faces = collapse_edges(vertices,faces,edges,priorities)
                
                #plot(src_vertices,src_faces,vertices,faces)
                
                self.assertTrue((calc_face_normals(vertices,faces[faces[:,0]>0])[:,2]>0).all().item())

        
if __name__ == '__main__':
    unittest.main()
