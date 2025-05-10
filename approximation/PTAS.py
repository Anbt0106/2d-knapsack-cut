import math
from typing import List, Tuple

try:
    import pulp
except ImportError:
    __import__('subprocess').check_call([__import__('sys').executable, '-m', 'pip', 'install', 'pulp'], stdout=None,
                                        stderr=None);
from collections import Counter
from itertools import combinations
import sys

sys.setrecursionlimit(20000)

# item indices
WIDTH = 0
HEIGHT = 1

from approximation.APTAS import APTAS




class PTAS(APTAS):
    def __init__(self, items: List[Tuple[float, float]], epsilon: float):
        self.items = items
        self.epsilon = epsilon
        self.T = [item for item in items if item[1] >= epsilon]
        self.W = [item for item in items if item[0] >= epsilon ** 4]
        self.thin_items = [item for item in items if item[0] < epsilon ** 4]

    def pack_tall_shelf(self, S_wide: List[Tuple[float, float]], H: float) -> Tuple[float, List[Tuple[float, float]]]:
        """
        Pack thin items into a tall shelf with wide items S_wide and height H.
        Returns: (profit, packed_thin) - Total profit and list of packed thin items.
        """
        w_S = sum(item[0] for item in S_wide)
        remaining_width = 1.0 - w_S
        thin_candidates = [item for item in self.thin_items if item[1] <= H]
        sorted_thin = sorted(thin_candidates, key=lambda x: -x[1])
        packed_thin = []
        current_width = 0.0

        if w_S >= 1 - self.epsilon ** 3:
            for item in sorted_thin:
                if current_width + item[0] <= remaining_width:
                    packed_thin.append(item)
                    current_width += item[0]
                elif remaining_width - current_width > 0:
                    frac = (remaining_width - current_width) / item[0]
                    packed_thin.append((item[0] * frac, item[1]))
                    break
        else:
            for item in sorted_thin:
                if current_width + item[0] <= remaining_width:
                    packed_thin.append(item)
                    current_width += item[0]
                elif remaining_width - current_width > 0:
                    frac = (remaining_width - current_width) / item[0]
                    packed_thin.append((item[0] * frac, item[1]))
                    break

        profit = sum(w * h for w, h in S_wide) + sum(w * h for w, h in packed_thin)
        return profit, packed_thin

    def solve(self) -> float:
        """
        Solve the 2SKP using PTAS, returning the maximum profit.
        """
        N_minus_T = [item for item in self.items if item not in self.T]
        if not self.nfdh_pack(N_minus_T, H=1.0):
            return sum(w * h for w, h in self.items) * (1 - self.epsilon)

        w_T = sum(item[0] for item in self.T)
        if w_T >= self.epsilon ** 3:
            accuracy = self.epsilon ** 2 * min(1, w_T)
            return sum(w * h for w, h in self.items) * (1 - accuracy)

        profit_N_minus_T = sum(w * h for w, h in N_minus_T)
        best_profit = profit_N_minus_T
        tall_heights = set(item[1] for item in self.T)

        max_wide = int(1 / (self.epsilon ** 4)) + 1
        for H in tall_heights:
            S_candidates = [item for item in self.W if item[1] <= H]
            for r in range(1, min(len(S_candidates) + 1, max_wide)):
                for S_wide_tuple in combinations(S_candidates, r):
                    S_wide = list(S_wide_tuple)
                    if sum(item[0] for item in S_wide) > 1:
                        continue
                    tall_profit, packed_thin = self.pack_tall_shelf(S_wide, H)

                    N_prime = [item for item in self.items if item not in S_wide and item not in packed_thin]
                    T_prime = [item for item in N_prime if item[1] >= self.epsilon * (1 - H)]
                    N_prime_minus_T_prime = [item for item in N_prime if item not in T_prime]

                    if not self.nfdh_pack(N_prime_minus_T_prime, H=1 - H):
                        lower_profit = sum(w * h for w, h in N_prime) * (1 - self.epsilon)
                    else:
                        w_T_prime = sum(item[0] for item in T_prime)
                        if w_T_prime >= self.epsilon ** 3:
                            accuracy = self.epsilon ** 2 * min(1, w_T_prime)
                            lower_profit = sum(w * h for w, h in N_prime) * (1 - accuracy)
                        else:
                            lower_profit = sum(w * h for w, h in N_prime_minus_T_prime)

                    total_profit = tall_profit + lower_profit
                    best_profit = max(best_profit, total_profit)

        return best_profit
