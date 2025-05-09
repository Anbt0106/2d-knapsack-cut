from approximation.PTAS import PTAS
import random
from typing import List, Tuple

class TestDriver:
    def __init__(self):
        pass

    def generate_test_case(self, num_items: int, width_range=(0.01, 1.0), height_range=(0.01, 1.0)) -> List[Tuple[float, float]]:
        items = []
        for _ in range(num_items):
            width = random.uniform(width_range[0], width_range[1])
            height = random.uniform(height_range[0], height_range[1])
            items.append((width, height))
        return items

    def testPTAS(self, num_cases=5, items_per_case=10):
        for i in range(num_cases):
            items = self.generate_test_case(items_per_case)
            solver = PTAS(items, epsilon=0.01)
            result = solver.solve()
            print(f"Case {i + 1}: approx profit = {result['profit']:.4f}")

