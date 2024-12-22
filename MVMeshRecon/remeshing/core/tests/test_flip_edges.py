import unittest
import torch
from core.remesh import calc_edges, calc_face_normals, flip_edges
import matplotlib.pyplot as plt
from core.tests.grid import area, make_grid

device='cuda'

def tensor(*args, **kwargs):
    return torch.tensor(*args, device=device, **kwargs)
    
def sort_faces(a:torch.Tensor):
    b = a.clone()
    for i in range(b.shape[0]):
        b[i] = b[i].roll(-b[i].argmin().item(),dims=[-1])
    for d in [2,1,0]:
        b = b[b[:,d].sort(stable=True)[1]]
    return b

def faces_equal(a:torch.Tensor,b:torch.Tensor):
    return sort_faces(a).equal(sort_faces(b))

def degree_loss(vertices:torch.Tensor,faces:torch.Tensor,with_border=True):
    V = vertices.shape[0]
    edges,_ = calc_edges(faces)
    E = edges.shape[0]
    vertex_degree = torch.zeros(V,dtype=torch.long,device=device) #V
    vertex_degree.scatter_(dim=0,index=edges.reshape(E*2),value=1,reduce='add')
    
    if with_border:
        x,y,_ = vertices.unbind(dim=-1)
        vertex_is_inside = (x>0) & (x<x[1:].max()) & (y>0) & (y<y[1:].max())
        goal = torch.where(vertex_is_inside,6,4)
    else:
        goal = 6

    return ((vertex_degree-goal)[1:]**2).sum().item()

def plot(vertices:torch.Tensor,src_faces:torch.Tensor,faces:torch.Tensor):
    def p(vertices,faces,shrink,color):
        f = vertices[faces] #F,3,3
        f = torch.lerp(f, f.mean(dim=1,keepdim=True), shrink)
        f = f.cpu().numpy()
        plt.fill(f[:,:,0].T,f[:,:,1].T, edgecolor=color, fill=False)
    p(vertices,src_faces,.05,'b')
    p(vertices,faces,.1,'r')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

class TestFlipEdges(unittest.TestCase):

    def test_flip_edges_simple(self):
        vertices,faces = make_grid(tensor([
                [0,0,0],
                [0,1,1],
                [0,0,1],
            ]).bool())
        
        src_faces = faces.clone()

        edges,face_to_edge,edge_to_face = calc_edges(faces, with_edge_to_face=True)
        flip_edges(vertices,faces,edges,edge_to_face,with_border=True,stable=True)
        #plot(vertices,src_faces,faces)

        _,faces_expected = make_grid(tensor([
                [0,0,0],
                [0,0,1],
                [0,0,1],
            ]).bool())

        self.assertTrue(faces_equal(faces,faces_expected))

    def test_flip_edges_normals(self):
        vertices,faces = make_grid(tensor([
                [0,0,0],
                [0,1,1],
                [0,0,1],
            ]).bool())
        vertices[7] = torch.tensor([1.2,1.8,0])
        
        src_faces = faces.clone()

        edges,face_to_edge,edge_to_face = calc_edges(faces, with_edge_to_face=True)
        flip_edges(vertices,faces,edges,edge_to_face,with_border=True,with_normal_check=True,stable=True)
        #plot(vertices,src_faces,faces)

        _,faces_expected = make_grid(tensor([
                [0,0,0],
                [0,1,1],
                [0,0,1],
            ]).bool())

        self.assertTrue(faces_equal(faces,faces_expected))

    def test_flip_edges_random(self):

        for i in range(10):
            for w in range(1,100):
                flip = torch.randint(0,2,(w,w),dtype=torch.bool,device=device)
                vertices,faces = make_grid(flip)
                edges,_,edge_to_face = calc_edges(faces, with_edge_to_face=True)

                self.assertEqual(area(vertices,faces),w*w)
                loss = degree_loss(vertices,faces)
                src_faces = faces.clone()

                flip_edges(vertices,faces,edges,edge_to_face)
                
                #plot(vertices,src_faces,faces)
                
                self.assertTrue((calc_face_normals(vertices,faces)[1:,2]>0).all().item())
                self.assertEqual(area(vertices,faces),w*w)
                self.assertLessEqual(degree_loss(vertices,faces),loss)


if __name__ == '__main__':
    unittest.main()
