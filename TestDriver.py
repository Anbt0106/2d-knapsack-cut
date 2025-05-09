from approximation.PTAS import PTAS
import random
from typing import List, Tuple
from dynamic_programming import DP
from dynamic_programming.DP import DPSolver
import random
import time
import sys

class TestDriver:
    def __init__(self):
        pass

    def generate_test_case(self, num_items: int, width_range=(0.01, 1.0), height_range=(0.01, 1.0)) -> List[
        Tuple[float, float]]:
        items = []
        for _ in range(num_items):
            width = random.uniform(width_range[0], width_range[1])
            height = random.uniform(height_range[0], height_range[1])
            items.append((width, height))
        return items

    def testPTAS(self, num_cases=5, items_per_case=100):
        for i in range(num_cases):
            items = self.generate_test_case(items_per_case)
            start_time = time.time()
            solver = PTAS(items, epsilon=0.01)
            result = solver.solve()
            end_time = time.time()

            memory_used = sys.getsizeof(solver)
            elapsed_time = end_time - start_time
            print(f"Case {i + 1}: approx profit = {result['profit']:.4f}, Time = {elapsed_time:.6f}s, Memory = {memory_used} bytes")

    def testDP(self, num_cases=5, items_per_case=100):
        for i in range(num_cases):
            items = self.generate_test_case(items_per_case)
            start_time = time.time()
            solver = DPSolver(items, resolution=100)  # Sử dụng DPSolver thay vì PTAS
            result = solver.solve()
            end_time = time.time()

            memory_used = sys.getsizeof(solver)

            elapsed_time = end_time - start_time
            print(f"Case {i + 1}: max profit = {result['profit']:.4f}, Time = {elapsed_time:.6f}s, Memory = {memory_used} bytes")
