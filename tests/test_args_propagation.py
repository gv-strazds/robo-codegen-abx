import unittest
from unittest.mock import MagicMock
import numpy as np

# item_generation has no isaacsim dependency; path setup is in conftest.py
from item_generation import ItemGenerator, PositionGenerator

# Define a mock position generator with capacity
class MockPosGen(PositionGenerator):
    def __init__(self, cap=None):
        self._capacity = cap
    
    @property
    def capacity(self):
        return self._capacity
        
    def get_positions(self, count, seed=None):
        return [np.zeros(3) for _ in range(count)]

class TestItemGenerator(unittest.TestCase):
    def test_min_only_finite_capacity(self):
        # Capacity = 10, range = (2, None) -> should result in count between 2 and 10
        pos_gen = MockPosGen(cap=10)
        gen = ItemGenerator(position_generator=pos_gen)
        
        # Test 100 times to ensure range
        for _ in range(100):
            items = gen.generate(count_range=(2, None))
            self.assertTrue(2 <= len(items) <= 10, f"Count {len(items)} not in range [2, 10]")

    def test_min_only_infinite_capacity(self):
        # Capacity = None, range = (2, None) -> currently falls back to fixed at min (2)
        pos_gen = MockPosGen(cap=None)
        gen = ItemGenerator(position_generator=pos_gen)
        
        for _ in range(100):
            items = gen.generate(count_range=(2, None))
            self.assertEqual(len(items), 2, f"Count {len(items)} should be 2 for infinite capacity fallback")

    def test_tuple_range(self):
        pos_gen = MockPosGen(cap=20)
        gen = ItemGenerator(position_generator=pos_gen)
        for _ in range(100):
            items = gen.generate(count_range=(3, 6))
            self.assertTrue(3 <= len(items) <= 6)


# Test Arg Parsing Logic (Simulated)
class TestArgParsing(unittest.TestCase):
    def resolve_counts(self, args):
        # Replicate logic from run_task.py
        pick_count = args.pick_count
        if pick_count is None:
            if args.pick_count_min is not None:
                if args.pick_count_max is not None:
                    pick_count = (args.pick_count_min, args.pick_count_max)
                else:
                    pick_count = (args.pick_count_min, None)
            elif args.pick_count_max is not None:
                pick_count = (1, args.pick_count_max)
                
        target_count = args.target_count
        if target_count is None:
            if args.target_count_min is not None:
                if args.target_count_max is not None:
                    target_count = (args.target_count_min, args.target_count_max)
                else:
                    target_count = (args.target_count_min, None)
            elif args.target_count_max is not None:
                target_count = (1, args.target_count_max)
        return pick_count, target_count

    def test_parsing(self):
        args = MagicMock()
        
        # Case 1: Pick Fixed
        args.pick_count = 5
        args.pick_count_min = None
        args.pick_count_max = None
        args.target_count = None
        args.target_count_min = None
        args.target_count_max = None
        
        p, t = self.resolve_counts(args)
        self.assertEqual(p, 5)
        self.assertIsNone(t)

        # Case 2: Pick Range
        args.pick_count = None
        args.pick_count_min = 2
        args.pick_count_max = 8
        p, t = self.resolve_counts(args)
        self.assertEqual(p, (2, 8))

        # Case 3: Pick Min Only
        args.pick_count = None
        args.pick_count_min = 3
        args.pick_count_max = None
        p, t = self.resolve_counts(args)
        self.assertEqual(p, (3, None))

        # Case 4: Max Only (New Fix)
        args.pick_count = None
        args.pick_count_min = None
        args.pick_count_max = 5
        p, t = self.resolve_counts(args)
        self.assertEqual(p, (1, 5))


if __name__ == '__main__':
    unittest.main()
