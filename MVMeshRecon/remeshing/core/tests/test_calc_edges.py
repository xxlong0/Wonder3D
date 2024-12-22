import unittest
import torch
from torch import nan
from core.remesh import calc_edge_length, calc_edges
import math

def tensor(*args, **kwargs):
    return torch.tensor(*args, **kwargs, device='cuda')
    

class TestCalcEdges(unittest.TestCase):

    def test_calc_edges(self):
        #  +y
        #
        #  v3--e5--v4      
        #  | \     |   
        #  |  \ f3 |
        #  e2  e3  e4     
        #  | f1 \  |    
        #  |     \ |
        #  v1--e1--v2     +x
        #
        vertices = tensor([
                [nan,nan,nan], #dummy
                [0,0,0],
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

        edges, face_to_edge, edge_to_face = calc_edges(faces,with_edge_to_face=True)

        edges_expected = tensor([
                [0,0], #dummy
                [1,2],
                [1,3],
                [2,3],
                [2,4],
                [3,4],
            ])        

        face_to_edge_expected =  tensor([
                [0,0,0], #dummy
                [1,3,2],
                [0,0,0],
                [4,5,3],
            ])

        edge_to_face_expected = tensor([
            [[0, 0],[0, 0]], #dummy
            [[1, 0],[0, 0]], #left: triangle 1, edge 0
            [[0, 0],[1, 2]], #right: triangle 1, edge 2
            [[1, 1],[3, 2]],
            [[3, 0],[0, 0]],
            [[0, 0],[3, 1]],
            ])
        
        self.assertTrue(edges.equal(edges_expected))
        self.assertTrue(face_to_edge.equal(face_to_edge_expected))
        self.assertTrue(edge_to_face.equal(edge_to_face_expected))

        length = calc_edge_length(vertices,edges)
        length_expected = tensor([nan,1,1,math.sqrt(2),1,1])
        self.assertTrue(length.allclose(length_expected,equal_nan=True))


if __name__ == '__main__':
    unittest.main()
