from typing import List, Tuple

class GuillotineDPSolver:
    """
    2D Knapsack with Guillotine Cuts via Dynamic Programming on a discretized grid.
    Container assumed to be a 1x1 square; resolution scales it to R x R integer grid.
    Items are floats (w,h) in (0,1]; we scale to integer sizes.
    """
    def __init__(self, items: List[Tuple[float, float]], resolution: int = 100):
        self.resolution = resolution
        # Scale items to integer grid
        self.items = []  # list of (wi, hi)
        for w, h in items:
            wi = int(w * resolution + 1e-6)
            hi = int(h * resolution + 1e-6)
            if wi > 0 and hi > 0 and wi <= resolution and hi <= resolution:
                self.items.append((wi, hi))
        # DP table: dp[w][h] = best area in w x h sub-rectangle
        self.dp = [[0] * (resolution + 1) for _ in range(resolution + 1)]

    def solve(self) -> dict:
        R = self.resolution
        dp = self.dp
        # Bottom-up fill
        for w in range(1, R + 1):
            for h in range(1, R + 1):
                best = 0
                # Try placing each item at top-left, then guillotine cut
                for wi, hi in self.items:
                    if wi <= w and hi <= h:
                        area = wi * hi
                        # cut off right strip after placing item
                        right_profit = dp[w - wi][h]
                        # cut off bottom strip after placing item
                        bottom_profit = dp[w][h - hi]
                        best = max(best,
                                   area + right_profit,
                                   area + bottom_profit)
                # Try vertical splits
                for x in range(1, w // 2 + 1):  # symmetry reduce half
                    comb = dp[x][h] + dp[w - x][h]
                    if comb > best:
                        best = comb
                # Try horizontal splits
                for y in range(1, h // 2 + 1):
                    comb = dp[w][y] + dp[w][h - y]
                    if comb > best:
                        best = comb
                dp[w][h] = best
        # Convert back to float area (divide by R^2)
        max_area = dp[R][R] / (R * R)
        return {'profit': max_area}