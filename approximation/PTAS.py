from typing import List, Tuple
from itertools import combinations
from approximation.APTAS import APTAS

HEIGHT, WIDTH = 1, 0


def nfdh_fits(items, max_height):
    row_height = 0
    current_row_width = 0
    for w, h in items:
        if current_row_width + w > 1:
            row_height += h
            current_row_width = w
        else:
            current_row_width += w
    row_height += items[-1][HEIGHT] if items else 0
    return row_height <= max_height


def fill_thin(shelves, thin_items):
    used_area = 0
    for shelf in shelves:
        width_left = shelf['width_left']
        for w, h in thin_items:
            if w <= width_left:
                used_area += w * h
                width_left -= w
    return used_area


class PTAS(APTAS):
    def __init__(self, items: List[Tuple[float, float]], epsilon: float):
        super().__init__(items, epsilon)

    def solve(self):
        def ptas2skp(items, eps, depth=0, max_depth=100):
            # Base case: stop if depth exceeded or few items remain
            if depth > max_depth or len(items) <= 1:
                return sum(w * h for w, h in items) if items else 0

            # Categorize items
            T = [j for j in items if j[HEIGHT] >= eps]  # Tall items
            W = [j for j in items if j[WIDTH] >= eps ** 4]  # Wide items
            short = [j for j in items if j not in T]  # Short items

            # Case 1: Short items don’t fit, increase eps (capped at 1.0)
            if not nfdh_fits(short, 1.0):
                new_eps = min(eps * 2, 1.0)
                return ptas2skp(items, new_eps, depth + 1, max_depth)

            # Case 2: Tall items’ width is large, decrease eps (floored at 0.01)
            wT = sum(j[WIDTH] for j in T)
            if wT >= eps ** 3:
                new_eps = max(eps ** 2 * min(1, wT), 0.01)
                return ptas2skp(items, new_eps, depth + 1, max_depth)

            # Case 3: Compute solution with shelves
            z1 = sum(w * h for w, h in short)  # Profit from short items
            z2 = 0
            for H in set(j[HEIGHT] for j in T):
                for r in range(1, min(len(W) + 1, 10)):  # Limit combinations
                    for S in combinations(W, r):
                        if sum(w for w, _ in S) <= 1 and max(h for _, h in S) <= H:
                            width_left = 1 - sum(w for w, _ in S)
                            thin_items = [j for j in items if j not in W]
                            thin_area = fill_thin([{'height': H, 'width_left': width_left}], thin_items)
                            used = set(S)
                            Np = [j for j in items if j not in used]
                            if Np and H < 1:
                                Np_scaled = [(w, h / (1 - H)) for w, h in Np]
                                z2p = ptas2skp(Np_scaled, eps, depth + 1, max_depth)
                            else:
                                z2p = 0
                            z2 = max(z2, z2p + sum(w * h for w, h in S) + thin_area)
            return max(z1, z2)

        return {'profit': ptas2skp(self.items, self.epsilon, depth=0)}
