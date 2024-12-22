import unittest
import torch
from torch import nan
from core.remesh import calc_vertex_normals,calc_face_normals,calc_face_ref_normals
import torch.nn.functional as tfunc
import math

device='cuda'

def tensor(*args, **kwargs):
    return torch.tensor(*args, device=device, **kwargs)
    

class TestCalcNormals(unittest.TestCase):

    def test_calc_normals(self):
        #          +z
        #          |
        #          4      
        #         /|\       
        #        / | \
        #       /  1  \
        #      / /   \ \
        #     3---------2 
        #   /             \ 
        # +x                +y

        vertices = tensor([
                [nan,nan,nan], #dummy
                [0,0,0],
                [0,1,0],
                [1,0,0],
                [0,0,1],
            ])   

        faces = tensor([
                [0,0,0], #dummy
                [1,2,3],
                [0,0,0], #unused
                [1,4,2],
                [1,3,4],
            ]) 

        V = vertices.shape[0]
        F = faces.shape[0]

        face_normals = calc_face_normals(vertices,faces,normalize=False)
        vertex_normals = calc_vertex_normals(vertices,faces,face_normals)
        ref_normals = calc_face_ref_normals(faces,vertex_normals,normalize=False)

        face_normals_expected = tensor([
                [nan,nan,nan], #dummy
                [0,0,-1], #n +z
                [nan,nan,nan], #unused
                [-1,0,0], #n +y
                [0,-1,0], #n +x 
            ])

        vertex_normals_expected = tfunc.normalize(tensor([
                [nan,nan,nan], #dummy
                [-1,-1,-1],
                [-1,0,-1],
                [0,-1,-1],
                [-1,-1,0],
            ]),dim=-1)
            
        l2 = 1/math.sqrt(2)
        l3 = 1/math.sqrt(3)
        ref_normals_expected = tensor([
                [nan,nan,nan], #dummy
                [-l3-l2,-l3-l2,-l3-2*l2],
                [nan,nan,nan], #unused
                [-l3-2*l2,-l3-l2,-l3-l2],
                [-l3-l2,-l3-2*l2,-l3-l2],
            ])

        self.assertTrue(face_normals.allclose(face_normals_expected,equal_nan=True))
        self.assertTrue(vertex_normals.allclose(vertex_normals_expected,equal_nan=True))
        self.assertTrue(ref_normals.allclose(ref_normals_expected,equal_nan=True))


if __name__ == '__main__':
    unittest.main()
