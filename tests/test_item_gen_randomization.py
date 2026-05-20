
import unittest
import numpy as np
from item_generation import GridPositionGenerator, CircularPositionGenerator, LayeredPositionGenerator

class TestItemGeneration(unittest.TestCase):

    def test_grid_randomization(self):
        # Create a 3x3 grid
        gen = GridPositionGenerator(center=np.array([0, 0, 0]), rows=3, cols=3, spacing_x=1.0, spacing_y=1.0, randomize=True)
        self.assertEqual(gen.capacity, 9)

        # Request 3 items
        count = 3
        positions1 = gen.get_positions(count, seed=42)
        positions2 = gen.get_positions(count, seed=43)

        # Verify count
        self.assertEqual(len(positions1), 3)
        self.assertEqual(len(positions2), 3)

        # Verify randomness (different seeds should produce different sets or orders)
        # Note: It's possible but unlikely that they produce the same set/order with different seeds for small samples.
        # But for 9 choose 3, there are 84 combinations * 3! permutations = 504 outcomes. Collision unlikley.
        
        # Check if lists are identical
        are_equal = True
        if len(positions1) == len(positions2):
            for i in range(len(positions1)):
                if not np.array_equal(positions1[i], positions2[i]):
                    are_equal = False
                    break
        self.assertFalse(are_equal, "Different seeds should produce different results for randomized grid")

        # Verify valid positions
        # All positions must lie on the grid points
        valid_coords = []
        center = np.array([0, 0, 0])
        # Grid is centered. 3x3 with spacing 1.
        # Xs: -1, 0, 1. Ys: -1, 0, 1. Z: 0.
        for r in range(3):
            for c in range(3):
                # Calculate expected coordinates manually
                # center is 0,0. width=2, length=2.
                # start_x = -1, start_y = -1
                x = -1 + c * 1.0
                y = -1 + r * 1.0
                valid_coords.append((x, y))
        
        for pos in positions1:
            found = False
            for vx, vy in valid_coords:
                if np.isclose(pos[0], vx) and np.isclose(pos[1], vy) and np.isclose(pos[2], 0):
                    found = True
                    break
            self.assertTrue(found, f"Position {pos} not found in valid grid points")

    def test_grid_sequential(self):
        # Create a 3x3 grid, randomize=False
        gen = GridPositionGenerator(center=np.array([0, 0, 0]), rows=3, cols=3, spacing_x=1.0, spacing_y=1.0, randomize=False)
        
        count = 3
        positions = gen.get_positions(count, seed=42)
        
        # Expected: top-left 3 items (row-major usually)
        # Based on implementation:
        # start_x = -1, start_y = -1.
        # Loops r in 0..2, c in 0..2
        # 1. r=0, c=0 -> (-1, -1)
        # 2. r=0, c=1 -> (0, -1)
        # 3. r=0, c=2 -> (1, -1)
        
        expected = [
            np.array([-1.0, -1.0, 0.0]),
            np.array([0.0, -1.0, 0.0]),
            np.array([1.0, -1.0, 0.0])
        ]
        
        self.assertEqual(len(positions), 3)
        for i in range(3):
            self.assertTrue(np.allclose(positions[i], expected[i]), f"Sequential failed at index {i}: {positions[i]} vs {expected[i]}")

    def test_circle_randomization(self):
        # Circle with capacity 12 (like a clock)
        gen = CircularPositionGenerator(center=np.array([0, 0, 0]), radius=1.0, count=12, randomize=True)
        
        count = 3
        positions1 = gen.get_positions(count, seed=1)
        positions2 = gen.get_positions(count, seed=2)
        
        self.assertEqual(len(positions1), 3)
        self.assertEqual(len(positions2), 3)
        
        # Verify randomness
        are_equal = True
        for i in range(len(positions1)):
            if not np.array_equal(positions1[i], positions2[i]):
                are_equal = False
                break
        self.assertFalse(are_equal, "Different seeds should produce different results for randomized circle")

        # Verify positions match valid slots (multiples of 30 deg)
        # 360 / 12 = 30 deg = PI/6
        for pos in positions1:
            # angle = atan2(y, x)
            angle = np.arctan2(pos[1], pos[0])
            # Normalize to 0..2PI
            if angle < 0: angle += 2*np.pi
            
            # Check if it's close to k * PI/6
            slot = angle / (np.pi / 6)
            closest_slot = round(slot)
            self.assertTrue(abs(slot - closest_slot) < 0.001, f"Angle {angle} does not align with slots. Slot {slot}")

    def test_circle_default(self):
        # Circle dynamic spacing (old behavior), random=False
        gen = CircularPositionGenerator(center=np.array([0, 0, 0]), radius=1.0, count=12, randomize=False)
        
        # Request 3, should get 0, 120, 240 degrees (even spacing for 3)
        count = 3
        positions = gen.get_positions(count) # seed ignored
        
        expected_angles_deg = [0, 120, 240]
        
        for i, pos in enumerate(positions):
            angle = np.arctan2(pos[1], pos[0])
            deg = np.degrees(angle)
            if deg < -0.1: deg += 360
            
            # Find closest match in expected (order is usually sequential 0..N)
            # Actually implementation is i in range(count): 2*pi*i/count
            # so 0, 120, 240 exactly.
            expected = expected_angles_deg[i]
            self.assertTrue(abs(deg - expected) < 0.1, f"Expected {expected}, got {deg}")

    # --- LayeredPositionGenerator tests ---

    def test_layered_grid_basic(self):
        """2x2 grid, 3 layers: verify Z values per layer."""
        base = GridPositionGenerator(
            center=np.array([0, 0, 0.1]), rows=2, cols=2,
            spacing_x=0.1, spacing_y=0.1, randomize=False
        )
        layer_h = 0.05
        gen = LayeredPositionGenerator(base, num_layers=3, layer_height=layer_h)

        self.assertEqual(gen.capacity, 12)  # 4 * 3
        positions = gen.get_positions(12)
        self.assertEqual(len(positions), 12)

        # Layer 0: z = 0.1, Layer 1: z = 0.15, Layer 2: z = 0.2
        for i in range(4):
            self.assertAlmostEqual(positions[i][2], 0.1, places=6)
        for i in range(4, 8):
            self.assertAlmostEqual(positions[i][2], 0.1 + layer_h, places=6)
        for i in range(8, 12):
            self.assertAlmostEqual(positions[i][2], 0.1 + 2 * layer_h, places=6)

    def test_layered_partial_top_layer(self):
        """7 items on 2x2 grid (cap=4) with 3 layers -> 4 + 3 + 0."""
        base = GridPositionGenerator(
            center=np.array([0, 0, 0]), rows=2, cols=2,
            spacing_x=0.1, spacing_y=0.1, randomize=False
        )
        gen = LayeredPositionGenerator(base, num_layers=3, layer_height=0.05)

        positions = gen.get_positions(7)
        self.assertEqual(len(positions), 7)

        # Layer 0: indices 0-3 (z=0)
        for i in range(4):
            self.assertAlmostEqual(positions[i][2], 0.0, places=6)
        # Layer 1: indices 4-6 (z=0.05), partial
        for i in range(4, 7):
            self.assertAlmostEqual(positions[i][2], 0.05, places=6)

    def test_layered_xy_consistency(self):
        """XY positions must match across layers."""
        base = GridPositionGenerator(
            center=np.array([1.0, 2.0, 0.5]), rows=2, cols=3,
            spacing_x=0.2, spacing_y=0.3, randomize=False
        )
        gen = LayeredPositionGenerator(base, num_layers=2, layer_height=0.1)
        positions = gen.get_positions(12)  # 6 per layer
        self.assertEqual(len(positions), 12)

        for i in range(6):
            # XY of layer 0 item i should match XY of layer 1 item i
            np.testing.assert_allclose(positions[i][:2], positions[i + 6][:2])
            # Z should differ by layer_height
            self.assertAlmostEqual(positions[i + 6][2] - positions[i][2], 0.1, places=6)

    def test_layered_circle(self):
        """Wrapping CircularPositionGenerator: verify capacity and Z offsets."""
        base = CircularPositionGenerator(
            center=np.array([0, 0, 0]), radius=1.0, count=6, randomize=False
        )
        gen = LayeredPositionGenerator(base, num_layers=2, layer_height=0.2)

        self.assertEqual(gen.capacity, 12)
        positions = gen.get_positions(12)
        self.assertEqual(len(positions), 12)

        # Layer 0: z=0, Layer 1: z=0.2
        for i in range(6):
            self.assertAlmostEqual(positions[i][2], 0.0, places=6)
        for i in range(6, 12):
            self.assertAlmostEqual(positions[i][2], 0.2, places=6)

        # XY should match between layers
        for i in range(6):
            np.testing.assert_allclose(positions[i][:2], positions[i + 6][:2])

    def test_layered_single_layer(self):
        """num_layers=1 should match base generator output exactly."""
        base = GridPositionGenerator(
            center=np.array([0, 0, 0]), rows=2, cols=2,
            spacing_x=0.1, spacing_y=0.1, randomize=False
        )
        gen = LayeredPositionGenerator(base, num_layers=1, layer_height=0.05)

        self.assertEqual(gen.capacity, 4)
        base_positions = base.get_positions(4)
        layered_positions = gen.get_positions(4)

        self.assertEqual(len(layered_positions), 4)
        for i in range(4):
            np.testing.assert_array_equal(base_positions[i], layered_positions[i])

    def test_layered_zero_count(self):
        """Requesting 0 items returns empty list."""
        base = GridPositionGenerator(
            center=np.array([0, 0, 0]), rows=2, cols=2,
            spacing_x=0.1, spacing_y=0.1, randomize=False
        )
        gen = LayeredPositionGenerator(base, num_layers=3, layer_height=0.05)
        positions = gen.get_positions(0)
        self.assertEqual(len(positions), 0)

    def test_layered_exceeds_capacity(self):
        """Requesting more than capacity caps at capacity."""
        base = GridPositionGenerator(
            center=np.array([0, 0, 0]), rows=2, cols=2,
            spacing_x=0.1, spacing_y=0.1, randomize=False
        )
        gen = LayeredPositionGenerator(base, num_layers=2, layer_height=0.05)
        self.assertEqual(gen.capacity, 8)

        positions = gen.get_positions(100)
        self.assertEqual(len(positions), 8)


if __name__ == '__main__':
    unittest.main()
