import unittest
import torch
from torch import nan
from core.remesh import calc_edge_length, calc_edges, calc_face_collapses, calc_face_normals, calc_vertex_normals

device='cuda'

def tensor(*args, **kwargs):
    return torch.tensor(*args, device=device, **kwargs)
    

class TesCalcFaceCollapses(unittest.TestCase):

    def test_face_flip(self):
        #  +y
        #        
        #  4-----5-----6  
        #  | \   |   (5) --> 5 moved to [1.9,0.8,0]
        #  |   \ | /   |
        #  1-----2-----3     +x
        #
        #  4-----------5 --> 5 and 6 collapsed to [1.95,0.9,0]
        #  | \       / | 
        #  |   \   /   | 
        #  1-----2-----3     +x

        vertices = tensor([
                [nan,nan,nan], #dummy
                [0.,0,0],
                [1,0,0],
                [2,0,0],
                [0,1,0],
                [1.9,0.8,0], #!
                [2,1,0],
            ])   

        faces = tensor([
                [0,0,0], #dummy
                [1,2,4],
                [0,0,0], #unused
                [2,5,4],
                [2,3,6],
                [2,6,5],
            ]) 

        face_normals = calc_face_normals(vertices,faces,normalize=False) #F,3
        vertex_normals = calc_vertex_normals(vertices,faces,face_normals) #V,3
        edges,face_to_edge = calc_edges(faces) #E,2 F,3
        edge_length = calc_edge_length(vertices,edges) #E
        collapses = calc_face_collapses(vertices,faces,edges,face_to_edge,edge_length,face_normals,vertex_normals,shortest_probability=1)

        collapses_expected = torch.zeros(edges.shape[0],dtype=torch.bool,device=device)
        collapses_expected[-1] = True

        self.assertTrue(collapses.equal(collapses_expected))

    def test_small_face(self):
        #  +y
        #        
        #  3--4
        #  | \ \ 
        #  |   \\
        #  1-----2

        vertices = tensor([
                [nan,nan,nan], #dummy
                [0.,0,0],
                [1,0,0],
                [0,1,0],
                [0.1,1,0],
            ])   

        faces = tensor([
                [0,0,0], #dummy
                [1,2,3],
                [2,4,3],
            ]) 

        face_normals = calc_face_normals(vertices,faces,normalize=False) #F,3
        vertex_normals = calc_vertex_normals(vertices,faces,face_normals) #V,3
        edges,face_to_edge = calc_edges(faces) #E,2 F,3
        edge_length = calc_edge_length(vertices,edges) #E
        min_edge_length = torch.full((vertices.shape[0],),fill_value=1.,device=device)
        collapses = calc_face_collapses(vertices,faces,edges,face_to_edge,edge_length,face_normals,vertex_normals,min_edge_length=min_edge_length,shortest_probability=1)

        collapses_expected = torch.zeros(edges.shape[0],dtype=torch.bool,device=device)
        collapses_expected[-1] = True
        self.assertTrue(collapses.equal(collapses_expected))

        # random edge selection
        for _ in range(200):
            collapses = calc_face_collapses(vertices,faces,edges,face_to_edge,edge_length,face_normals,vertex_normals,min_edge_length=min_edge_length,shortest_probability=0.7)
            collapses_expected = torch.zeros([3,edges.shape[0]],dtype=torch.bool,device=device)
            collapses_expected[0,3] = True
            collapses_expected[1,4] = True
            collapses_expected[2,5] = True
            self.assertEqual((collapses==collapses_expected).all(dim=-1).sum().item(),1)


if __name__ == '__main__':
    unittest.main()
