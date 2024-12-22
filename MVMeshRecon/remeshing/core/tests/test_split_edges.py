import unittest
from core.remesh import calc_edges, calc_face_normals, split_edges
import torch
from torch import nan

device='cuda'

def tensor(*args, **kwargs):
    return torch.tensor(*args, device=device, **kwargs)
    
class TestSplitEdges(unittest.TestCase):

    def test_split_edges(self):
        #  +y
        #
        #  3             3          
        #  |\            |\         
        #  | \           | 6           
        #  |  \          |/ \      
        #  1---2         1---2     +x
        #  |  /          |/ /     
        #  | /           5-7          
        #  |/            |/         
        #  4             4                
        #
        faces = tensor(dtype=torch.long,
            data=[
                [0,0,0], #dummy
                [1,2,3],
                [0,0,0], #unused
                [1,4,2],
            ])
        vertices = tensor([
                [nan,nan,nan], #dummy
                [0,0,0],
                [1,0,0],
                [0,1,0],
                [0,-1,0],
            ])        
        faces_expected = tensor(dtype=torch.long,
            data=[
                [0,0,0], #dummy
                [1,6,3],
                [5,7,2],
                [2,6,1],
                [1,5,2],
                [4,7,5],
            ])
        vertices_expected = tensor([
                [nan,nan,nan], #dummy
                [0,0,0],
                [1,0,0],
                [0,1,0],
                [0,-1,0],
                [0,-.5,0],
                [.5,.5,0],
                [.5,-.5,0],
            ])

        edges,face_to_edge = calc_edges(faces)

        splits = torch.zeros(edges.shape[0],dtype=torch.bool,device=device)
        splits[[3,4,5]] = True 
        self.assertTrue(edges[[3,4,5]].equal(tensor([[1,4],[2,3],[2,4]])))

        vertices,faces = split_edges(vertices,faces,edges,face_to_edge,splits)

        self.assertTrue(faces.equal(faces_expected))
        self.assertTrue(vertices.allclose(vertices_expected,equal_nan=True))

    def test_split_edges_random(self):

        for _ in range(100):

            faces = tensor(dtype=torch.long,
                data=[
                    [0,0,0], #dummy
                    [1,2,3],
                ])

            vertices = tensor([
                    [nan,nan,nan], #dummy
                    [0,0,0],
                    [1,0,0],
                    [0,1,0],
                ])        

            for _ in range(10):
                area = calc_face_normals(vertices,faces)[1:,2].sum().item()/2
                self.assertAlmostEqual(area,.5)
                edges,face_to_edge = calc_edges(faces)
                splits = torch.randint(0,3,size=(edges.shape[0],),device=device)==0
                splits[0] = False #dont split dummy
                vertices,faces = split_edges(vertices,faces,edges,face_to_edge,splits)


if __name__ == '__main__':
    unittest.main()
