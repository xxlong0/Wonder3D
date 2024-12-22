import unittest
import torch
from util.func import laplacian

class TestLaplacian(unittest.TestCase):

    def test_laplacian(self):
        edges = torch.tensor([[0,1],[0,2]],dtype=torch.int32)
        L = laplacian(3, edges)
        expected = torch.tensor([[2,-1,-1],[-1,1,0],[-1,0,1]],dtype=torch.float32)
        self.assertTrue(torch.allclose(L.to_dense(),expected))

if __name__ == '__main__':
    unittest.main()
