import math
from typing import List, Tuple

try:
    import pulp
except ImportError:
    __import__('subprocess').check_call([__import__('sys').executable, '-m', 'pip', 'install', 'pulp'], stdout=None,
                                        stderr=None); import pulp
from collections import Counter
from itertools import combinations
import sys

sys.setrecursionlimit(20000)

HEIGHT, WIDTH = 1, 0


class APTAS:
    aleph = None

    def __init__(self, items: List[Tuple[float, float]], epsilon: float):
        self.items = items
        self.epsilon = epsilon

    # part a
    def decomposition(self):
        """Find optimal f and group items into A_i sets.
       Decomposition:

   (a.1) Guess the integer ff (1≤f≤t:=1/ε) and for i∈I ⁣ ⁣Ni∈IN remove all the items
       with height in (εf+it,εf+it−1).
   (a.2) Let AiAi​ be the set of the items with heights in [εf+it−1,εf+(i−1)t] for each
       i∈I ⁣ ⁣Ni∈IN. Form the shelves in the final solution by considering separately each set Ai​.
       """
        t = math.ceil(1 / self.epsilon)
        best_profit = 0
        opt_f = None
        opt_A = []

        # guess f by bruteforcing
        for f in range(1, t + 1):
            A = [[] for _ in range(t)]
            for item in self.items:
                h = item[HEIGHT]
                for i in range(t):
                    lower = self.epsilon ** (f + i * t - 1)
                    upper = self.epsilon ** t if i == 0 else self.epsilon ** (f + (i - 1) * t)
                    if lower <= h <= upper:
                        A[i].append(item)
                        break

            profits = [sum(w * h for w, h in A_i) for A_i in A]
            if not profits:
                continue
            # find the idx of the smallest profit
            min_idx = profits.index(min(p for p in profits if p > 0) if any(p > 0 for p in profits) else 0)
            current_A = [A_i for idx, A_i in enumerate(A) if idx != min_idx]  # remove height group with smallest profit
            current_profit = sum(w * h for A_i in current_A for w, h in A_i)
            if current_profit > best_profit:  # take f that results in highest profit
                best_profit = current_profit
                opt_f = f
                opt_A = current_A

        return opt_f, opt_A

    # part b
    def subdivide_and_melt(self, f: int, A: List[List[Tuple[float, float]]]):
        """Divide items into wide and thin; return thin items separately.
       Item Subdivision and Vertical Melting: For each set Ai​, i∈I ⁣ ⁣Ni∈IN:
   (b.1) Separate wide items (having width at least εtεt) from thin items (having width smaller than εtεt).
   (b.2) Allow thin items to be cut vertically.
       """
        t = math.ceil(1 / self.epsilon)
        wide_groups = []
        thin_items_groups = []
        for A_i in A:
            wide_items = [item for item in A_i if item[WIDTH] >= self.epsilon ** t]
            thin_items = [item for item in A_i if item[WIDTH] < self.epsilon ** t]
            wide_groups.append(wide_items)
            thin_items_groups.append(thin_items)
        return wide_groups, thin_items_groups

    # part c
    def low_shelves(self, wide_groups: List[List[Tuple[float, float]]],
                    thin_items_groups: List[List[Tuple[float, float]]], f: int):
        """Definition of Low Shelves: For each set AiAi​, i∈I ⁣ ⁣N∖{0}i∈IN∖{0}:
   (c.1) Round-up all heights to the nearest multiple of εf+itεf+it, letting h‾1,…,h‾gih1​,…,hgi​​ be
       the distinct heights after rounding and n1,…,ngin1​,…,ngi​​ the numbers of items of each height. For the profit (area) computation, consider rounded-down heights.
   (c.2) For each vector (m1,…,mgi)(m1​,…,mgi​​) such that 0≤mj≤nj (j=1,…,gi)0≤mj​≤nj​(j=1,…,gi​) and
       H:=∑j=1gimjh‾j≤1H:=∑j=1gi​​mj​hj​≤1, if not all items in AiAi​ can be packed in shelves of total height at most HH by NFDH:
       (c.2.1) Preprocess the wide items in AiAi​.
       (c.2.2) Round the widths of the preprocessed items with the same height after Step (c.1) by
               linear grouping. For the profit (area) computation, consider rounded-down widths.
       (c.2.3) Pack some items in AiAi​ in shelves of overall height at most HH and overall maximum
               profit by enumeration of the packings of wide items in AiAi​ combined with the solution of (1)-(6)."""

        t = math.ceil(1 / self.epsilon)
        pi_list = []
        for i in range(1, len(wide_groups)):
            wide = wide_groups[i]
            thin = thin_items_groups[i]
            if not wide and not thin:
                pi_list.append({})
                continue

            # Round wide item heights
            heights_info = [(w, h, math.ceil(h / self.epsilon ** (f + i * t)) * self.epsilon ** (f + i * t)) for w, h in
                            wide]
            hc_list = [hc for _, _, hc in heights_info]
            counts_map = Counter(hc_list)
            heights_i = sorted(counts_map.keys(), reverse=True)
            counts_i = [counts_map[hc] for hc in heights_i]

            def enumerate_m(heights, counts, max_height=1.0):
                g = len(heights)
                current = [0] * g

                def dfs(j, remaining_height):
                    if j == g:
                        yield tuple(current)
                        return
                    max_mj = min(counts[j], int(remaining_height // heights[j]))
                    for mj in range(max_mj + 1):
                        current[j] = mj
                        yield from dfs(j + 1, remaining_height - mj * heights[j])
                    current[j] = 0

                yield from dfs(0, max_height)

            pi = {}
            for m_vec in enumerate_m(heights_i, counts_i):
                item_list = []
                for idx, m_j in enumerate(m_vec):
                    h_j = heights_i[idx]
                    count = 0
                    for w, _, hc in heights_info:
                        if hc == h_j and count < m_j:
                            item_list.append((w, hc))
                            count += 1
                total_height = sum(m_j * heights_i[idx] for idx, m_j in enumerate(m_vec))
                if not self.nfdh_fits(item_list):
                    continue

                # MIP for wide and thin items
                prob = pulp.LpProblem("ShelfPacking", pulp.LpMaximize)
                x = pulp.LpVariable.dicts("x", range(len(wide)), cat='Binary')
                thin_area = pulp.LpVariable("thin_area", lowBound=0)
                prob += pulp.lpSum([w * h * x[j] for j, (w, h, _) in enumerate(heights_info)]) + thin_area  # Objective
                prob += pulp.lpSum(
                    [w * x[j] for j, (w, _, _) in enumerate(heights_info)]) + thin_area <= 1.0  # Width constraint
                prob += thin_area <= sum(w * h for w, h in thin)  # Thin area limit
                prob.solve(pulp.PULP_CBC_CMD(msg=0))
                if prob.status == pulp.LpStatusOptimal:
                    profit = pulp.value(prob.objective)
                    pi[total_height] = max(pi.get(total_height, 0), profit)
            pi_list.append(pi)
        return pi_list

    # part d
    def high_shelves(self, A0_shelf_configs, thin_items_groups, pi_list):
        """Combine A0 shelves with lower shelves."""
        best = {'profit': 0.0, 'shelf_config': None, 'thin_fill': 0.0, 'mc_profit': 0.0}
        A0_thin_items = thin_items_groups[0]
        for shelves in A0_shelf_configs[0]:
            for s in [shelves]:
                s['width_left'] = 1.0 - s['width_used']
            thin_area = fill_thin([s.copy() for s in [shelves]], A0_thin_items.copy())
            wide_profit = shelves['profit']
            used_height = shelves['height']
            residual = 1.0 - used_height
            mc_profit = multiple_choice_knapsack(pi_list, residual)
            total = wide_profit + thin_area + mc_profit
            if total > best['profit']:
                best.update({'profit': total, 'shelf_config': shelves, 'thin_fill': thin_area, 'mc_profit': mc_profit})
        return best

    def solve(self):
        f, A_groups = self.decomposition()
        if not f:
            return {'profit': 0.0}
        wide_groups, thin_items_groups = self.subdivide_and_melt(f, A_groups)
        pi_list = self.low_shelves(wide_groups, thin_items_groups, f)
        A0_shelf_configs = self.enumerate_A0_shelves(wide_groups[0])
        result = self.high_shelves(A0_shelf_configs, thin_items_groups, pi_list)
        return result

    def enumerate_A0_shelves(self, wide_items_A0: List[Tuple[float, float]]):
        """Enumerate shelf configurations for A0 wide items."""
        shelf_configs = []
        for r in range(1, min(len(wide_items_A0) + 1, 10)):  # Limit for efficiency
            for subset in combinations(wide_items_A0, r):
                total_width = sum(item[WIDTH] for item in subset)
                if total_width <= 1:
                    shelf_height = max(item[HEIGHT] for item in subset)
                    profit = sum(item[WIDTH] * item[HEIGHT] for item in subset)
                    shelf_configs.append({'height': shelf_height, 'width_used': total_width, 'profit': profit})
        return [shelf_configs]


# utility functions
def nfdh_fits(items: List[Tuple[float, float]], height_budget: float = 1.0) -> bool:
    return nfdh_height(items) <= height_budget


def nfdh_height(items: List[Tuple[float, float]], width_limit: float = 1.0) -> float:
    sorted_items = sorted(items, key=lambda x: x[HEIGHT], reverse=True)
    total_height = 0.0
    shelf_width = 0.0
    shelf_height = 0.0
    for w, h in sorted_items:
        if shelf_width + w <= width_limit:
            shelf_width += w
            shelf_height = max(shelf_height, h)
        else:
            total_height += shelf_height
            shelf_width = w
            shelf_height = h
    total_height += shelf_height
    return total_height


def fill_thin(shelves, thin_items):
    open_shelves = sorted(shelves, key=lambda s: -s['height'])
    thin = sorted(thin_items, key=lambda x: -x[HEIGHT])
    total_area = 0.0
    idx_thin = 0
    while open_shelves and idx_thin < len(thin):
        shelf = open_shelves[0]
        while idx_thin < len(thin) and thin[idx_thin][HEIGHT] > shelf['height']:
            idx_thin += 1
        if idx_thin == len(thin):
            break
        w, h = thin[idx_thin]
        if w <= shelf['width_left']:
            total_area += w * h
            shelf['width_left'] -= w
            idx_thin += 1
        else:
            frac_w = shelf['width_left']
            total_area += frac_w * h
            thin[idx_thin] = (w - frac_w, h)
            open_shelves.pop(0)
    return total_area


def multiple_choice_knapsack(pi_list, capacity):
    dp = {0.0: 0.0}
    for options in pi_list:
        new_dp = {}
        for used_h, used_p in dp.items():
            for h, p in options.items():
                nh = used_h + h
                if nh <= capacity:
                    new_dp[nh] = max(new_dp.get(nh, 0.0), used_p + p)
        dp = new_dp
    return max(dp.values(), default=0.0)
