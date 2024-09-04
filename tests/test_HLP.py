import unittest
import torch
from hierarchical_label_propagation import HLP

class TestHLP(unittest.TestCase):

    def setUp(self):
        self.HLP_lookup_table = torch.tensor([
            [1, 0, 0],
            [1, 1, 0],
            [0, 0, 1]
        ])
        self.hlp = HLP(self.HLP_lookup_table)

        self.hlp_default = HLP()

    def test_shape(self):
        self.assertEqual(self.hlp.shape, torch.Size([3, 3]))

    def test_default_shape(self):
        self.assertEqual(self.hlp_default.shape, torch.Size([527, 527]))

    def test_propagate_batch(self):
        result = self.hlp.propagate(torch.tensor([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1]
        ]))
        expected = torch.tensor([
            [0, 0, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1]
        ])
        self.assertTrue(torch.equal(result, expected))

    def test_propagate_single(self):
        test_cases = [
            (torch.tensor([0, 0, 0]), torch.tensor([0, 0, 0])),
            (torch.tensor([0, 0, 1]), torch.tensor([0, 0, 1])),
            (torch.tensor([0, 1, 0]), torch.tensor([1, 1, 0])),
            (torch.tensor([0, 1, 1]), torch.tensor([1, 1, 1])),
            (torch.tensor([1, 0, 0]), torch.tensor([1, 0, 0])),
            (torch.tensor([1, 0, 1]), torch.tensor([1, 0, 1])),
            (torch.tensor([1, 1, 0]), torch.tensor([1, 1, 0])),
            (torch.tensor([1, 1, 1]), torch.tensor([1, 1, 1])),
        ]

        for input_tensor, expected_tensor in test_cases:
            with self.subTest(input_tensor=input_tensor):
                result = self.hlp.propagate(input_tensor)
                print(result, expected_tensor)
                self.assertTrue(torch.equal(result, expected_tensor))


if __name__ == '__main__':
    unittest.main()
