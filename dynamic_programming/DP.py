
from typing import List, Tuple

class DPSolver:
    def __init__(self, items: List[Tuple[float, float]], resolution: int = 100):
        self.items = items
        self.resolution = resolution

    def solve(self) -> dict:
        W = self.resolution
        scaled_items = []
        for w, h in self.items:
            iw = int(w * W + 1e-6)
            if iw <= W:
                scaled_items.append((iw, h))

        n = len(scaled_items)
        dp = [0.0] * (W + 1)

        for iw, h in scaled_items:
            for w in range(W, iw - 1, -1):
                dp[w] = max(dp[w], dp[w - iw] + (iw / W) * h)
        return {'profit': max(dp)}
