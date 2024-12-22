import unittest
import torch
from torch import nan
from core.remesh import pack

def tensor(*args, **kwargs):
    return torch.tensor(*args, device='cuda', **kwargs)
    

class TestPack(unittest.TestCase):

    def test_pack(self):
        vertices = tensor([
                [nan,nan,nan], #dummy
                [0.,0,0],
                [1,0,0],
                [nan,nan,nan], #unused
                [0,1,0],
            ])   

        faces = tensor([
                [0,0,0], #dummy
                [1,2,4],
                [0,0,0], #unused
            ]) 

        vertices,faces = pack(vertices,faces)

        vertices_expected = tensor([
                [nan,nan,nan], #dummy
                [0.,0,0],
                [1,0,0],
                [0,1,0],
            ])   

        faces_expected = tensor([
                [0,0,0], #dummy
                [1,2,3],
            ]) 

        self.assertTrue(vertices.allclose(vertices_expected,equal_nan=True))
        self.assertTrue(faces.equal(faces_expected))

if __name__ == '__main__':
    unittest.main()
