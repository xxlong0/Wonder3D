import unittest
import torch
from torch import nan
from core.remesh import prepend_dummies, remove_dummies

def tensor(*args, **kwargs):
    return torch.tensor(*args, device='cuda', **kwargs)
    

class TestPack(unittest.TestCase):

    def test_pack(self):
        src_vertices = tensor([
                [0.,0,0],
                [1,0,0],
                [nan,nan,nan], #unused
                [0,1,0],
            ])   

        src_faces = tensor([
                [1,2,4],
            ]) 

        vertices,faces = prepend_dummies(src_vertices,src_faces)

        vertices_expected = tensor([
                [nan,nan,nan], #dummy
                [0.,0,0],
                [1,0,0],
                [nan,nan,nan], #unused
                [0,1,0],
            ])   

        faces_expected = tensor([
                [0,0,0], #dummy
                [2,3,5],
            ]) 

        self.assertTrue(vertices.allclose(vertices_expected,equal_nan=True))
        self.assertTrue(faces.equal(faces_expected))

        vertices,faces = remove_dummies(vertices,faces)

        self.assertTrue(vertices.allclose(src_vertices,equal_nan=True))
        self.assertTrue(faces.equal(src_faces))

if __name__ == '__main__':
    unittest.main()
