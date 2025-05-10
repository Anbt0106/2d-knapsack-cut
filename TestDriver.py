import random
import time
import tracemalloc
from typing import List, Tuple

from approximation.PTAS import PTAS
from dynamic_programming.DP import GuillotineDPSolver
from heuristic.TwoDKEDA import TwoDKEDA


class TestDriver:
    @staticmethod
    def generate_test_case(num_items, width_range=(0.01, 1.0), height_range=(0.01, 1.0)):
        items = []
        for _ in range(num_items):
            width = random.uniform(width_range[0], width_range[1])
            height = random.uniform(height_range[0], height_range[1])
            items.append((width, height))
        return items

    def test_ptas(self, items: List[Tuple[float, float]], epsilon: float) -> dict:
        """
        Run the PTAS solver on items.
        Returns a dict with profit, time in ms, and peak memory in bytes.
        """
        tracemalloc.start()
        start = time.perf_counter()
        solver = PTAS(items, epsilon)
        profit = solver.solve()
        elapsed = (time.perf_counter() - start) * 1000  # ms
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return {
            'profit': profit,
            'time_ms': elapsed,
            'mem_bytes': peak
        }

    def test_dp(self, items: List[Tuple[float, float]], resolution: int) -> dict:
        """
        Run the DP Guillotine solver on items.
        Returns a dict with profit, time in ms, and peak memory in bytes.
        """
        tracemalloc.start()
        start = time.perf_counter()
        solver = GuillotineDPSolver(items, resolution=resolution)
        result = solver.solve()
        elapsed = (time.perf_counter() - start) * 1000  # ms
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return {
            'profit': result['profit'],
            'time_ms': elapsed,
            'mem_bytes': peak
        }

    def test_2DKEDA(self, items: List[Tuple[float, float]], max_time: float = 10.0,
                    pop_size: int = 20, keep_frac: float = 0.2, alpha: float = 0.3) -> dict:
        """
        Run the TwoDKEDA solver on items.
        Returns a dict with profit, time in ms, and peak memory in bytes.
        """
        tracemalloc.start()
        start = time.perf_counter()
        solver = TwoDKEDA(items, max_time=max_time, pop_size=pop_size,
                          keep_frac=keep_frac, alpha=alpha)
        result = solver.evolve()
        elapsed = (time.perf_counter() - start) * 1000  # ms
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return {
            'profit': result['profit'],
            'time_ms': elapsed,
            'mem_bytes': peak
        }

    def run_experiments(self,
                        item_sizes: List[int],
                        epsilon: float = 0.05,
                        resolution: int = 100,
                        eda_time: float = 10.0) -> List[dict]:
        """
        Run comparative experiments across PTAS, DP, and EDA.
        Returns a list of result dicts.
        """
        results = []
        for n in item_sizes:
            print(f"\nTesting with {n} items...")
            items = self.generate_test_case(n)

            ptas_res = self.test_ptas(items, epsilon)
            print(
                f"PTAS: profit={ptas_res['profit']:.4f}, time={ptas_res['time_ms']:.2f}ms, mem={ptas_res['mem_bytes']} Bytes")

            dp_res = self.test_dp(items, resolution)
            print(
                f"DP:   profit={dp_res['profit']:.4f}, time={dp_res['time_ms']:.2f}ms, mem={dp_res['mem_bytes']} Bytes")

            eda_res = self.test_2DKEDA(items, max_time=eda_time)
            print(
                f"2DKEDA:  profit={eda_res['profit']:.4f}, time={eda_res['time_ms']:.2f}ms, mem={eda_res['mem_bytes']} Bytes")

            results.append({
                'n_items': n,
                'ptas': ptas_res,
                'dp': dp_res,
                'eda': eda_res
            })
        return results


if __name__ == '__main__':
    driver = TestDriver()
    # Define problem sizes and parameters
    sizes = [10, 20, 50, 100, 200, 500, 1000, 1500, 2000, 2500, 3000, 4500, 5000]
    eps = 0.05
    res = 100
    eda_t = 10.0  # seconds per run
    # Run experiments
    all_results = driver.run_experiments(sizes, epsilon=eps, resolution=res, eda_time=eda_t)

    import csv

    csv_filename = 'experiment_results.csv'
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Header row
        writer.writerow([
            'n_items',
            'PTAS_profit', 'PTAS_time_ms', 'PTAS_mem_B',
            'DP_profit', 'DP_time_ms', 'DP_mem_B',
            '2DKEDA_profit', '2DKEDA_time_ms', '2DKEDA_mem_B'
        ])
        # Data rows
        for r in all_results:
            writer.writerow([
                r['n_items'],
                f"{r['ptas']['profit']:.4f}", f"{r['ptas']['time_ms']:.2f}", r['ptas']['mem_bytes'],
                f"{r['dp']['profit']:.4f}", f"{r['dp']['time_ms']:.2f}", r['dp']['mem_bytes'],
                f"{r['2DKEDA']['profit']:.4f}", f"{r['2DKEDA']['time_ms']:.2f}", r['2DKEDA']['mem_bytes'],
            ])
    print(f"\nAll experiments completed. Results saved to {csv_filename}")
