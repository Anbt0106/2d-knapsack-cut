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
import itertools

sys.setrecursionlimit(20000)

# Item indices
WIDTH = 0
HEIGHT = 1


class APTAS:
  aleph = None

  def __init__(self, items: List[Tuple[float, float]], epsilon: float):
    self.items = items
    if epsilon > 0.1:
      self.epsilon = epsilon
    else:
      self.epsilon = 0.1

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
    best_profit = -1
    opt_f = None
    opt_A = []
    # print(t, self.epsilon)
    # guess f by bruteforcing all over the place
    for f in range(1, t + 1):
      A = [[] for _ in range(t)]
      for item in self.items:
        h = item[HEIGHT]

        for i in range(t):
          lower = self.epsilon ** (f + i * t - 1)
          upper = self.epsilon ** (f + (i - 1) * t) if not i == 0 else 1
          # print(lower, upper)
          if lower <= h <= upper:
            A[i].append(item)
            break

      profits = [sum(w * h for w, h in group) for group in A]

      # min_idx = profits.index(min(profits))  # rm the least profittable bucket

      # remaining_groups = [g for idx, g in enumerate(A) if idx != min_idx]
      # current_profit = sum(w * h for g in remaining_groups for w, h in g)
      current_profit = sum(profits)

      # print(opt_f)
      if current_profit > best_profit:
        best_profit = current_profit
        opt_f = f
        opt_A = A

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
  def nfdh_pack(self, items: List[Tuple[float, float]], H: float = 1.0) -> bool:
    """Check if NFDH can pack all items within height H."""
    sorted_items = sorted(items, key=lambda x: -x[1])  # Sort by decreasing height
    shelves = []
    current_shelf_items = []
    current_shelf_width = 0
    total_height = 0

    for item in sorted_items:
      if current_shelf_width + item[0] > 1:  # exceeding shelve's width
        if current_shelf_items:
          shelf_height = max(item[1] for item in current_shelf_items)
          if total_height + shelf_height > H:  # false if height exceeds H
            return False
          # otherwise
          shelves.append(current_shelf_items)
          total_height += shelf_height
        current_shelf_items = [item]
        current_shelf_width = item[0]
      else:
        current_shelf_items.append(item)
        current_shelf_width += item[0]

    if current_shelf_items:
      shelf_height = max(item[1] for item in current_shelf_items)
      if total_height + shelf_height > H:
        return False
      shelves.append(current_shelf_items)
    return True

  def low_shelves(self, wide_groups: List[List[Tuple[float, float]]],
                  thin_items_groups: List[List[Tuple[float, float]]], f: int):
    t = math.ceil(1 / self.epsilon)
    pi_list = []
    for i in range(1, len(wide_groups)):
      wide = wide_groups[i]
      thin = thin_items_groups[i]
      if not wide and not thin:
        pi_list.append({})
        continue

      # round heights to format (org_w, org_h, rounded_up_h, rounded_down_h)
      heights_info = [
        (w, h, math.ceil(h / self.epsilon ** (f + i * t)) * self.epsilon ** (f + i * t),
         math.floor(h / self.epsilon ** (f + i * t)) * self.epsilon ** (f + i * t))
        for w, h in wide
      ]

      hc_list = [hc for _, _, hc, _ in heights_info]
      counts_map = Counter(hc_list)
      heights_i = sorted(counts_map.keys(), reverse=True)  # distinct heights at shelf A_i (^h_g_i)
      counts_i = [counts_map[hc] for hc in heights_i]  # number of items at distinct height at shelf A_i (n_g_i)

      def enumerate_m(heights, counts, max_height=1.0):
        g = len(heights)  # number of distinct heights
        current = [0] * g  # current setting (m_1, m_2, ..., m_g_i)

        def dfs(j, remaining_height):
          if j == g:  # base case j == g: setting fully init-ed
            yield tuple(current)
            return

          # max value of m_j satisfies not exceeding number of items at that height and
          # still fits under remaining height budget
          max_mj = min(counts[j], int(remaining_height // heights[j]))
          # loop over all possibile m_j
          for mj in range(max_mj + 1):
            current[j] = mj
            yield from dfs(j + 1, remaining_height - mj * heights[j])
          current[j] = 0

        yield from dfs(0, max_height)

      pi = {}
      for m_vec in enumerate_m(heights_i, counts_i):
        # form the item list based on configuration m_j
        item_list = []
        for idx, m_j in enumerate(m_vec):
          h_j = heights_i[idx]
          count = 0
          for w, _, hc, _ in heights_info:
            if hc == h_j and count < m_j:
              item_list.append((w, hc))
              count += 1
        total_height = sum(m_j * heights_i[idx] for idx, m_j in enumerate(m_vec))  # H

        if self.nfdh_pack(item_list, total_height):
          # case 1: nfdh fits, compute profit directly, no profit loss
          profit = sum(w * h for w, h in item_list)
          pi[total_height] = max(pi.get(total_height, 0), profit)
        else:
          # case 2: nfdg doesn't fit, proceed with preprocessing, grouping, and MIP
          # (c.2.1) Preprocess wide items for each height class
          preprocessed_items = []  # R_ij
          H_limit = total_height
          limit = H_limit / (self.epsilon ** (f + (i + 1) * t - 1))

          for h_j in heights_i:
            S_ij = [item for item in heights_info if item[2] == h_j]
            T_ijk = {}
            for item in S_ij:
              w = item[0]
              k = math.ceil(w / self.epsilon ** (t + 1))
              if k not in T_ijk:
                T_ijk[k] = []
              T_ijk[k].append(item)
            for k in T_ijk:
              lower = 1 / self.epsilon
              upper = 1 / self.epsilon ** (t + 1)
              if k >= lower and k <= upper:
                T_ijk[k].sort(key=lambda x: x[0])
                selected = T_ijk[k][:int(limit)]
                preprocessed_items.extend(selected)

          # (c.2.2) linear Grouping
          # group by heights
          height_groups = {}
          for item in preprocessed_items:
            hc = item[2]
            if hc not in height_groups:
              height_groups[hc] = []
            height_groups[hc].append(item)

          grouped_items = []  # format (min_width, height, hc, hf)
          threshold = 8 / (self.epsilon ** (5 * t))
          for hc, items in height_groups.items():
            r_ij = len(items)
            if r_ij < threshold:
              grouped_items.extend([(w, w, h, hc) for w, h, hc, _ in items])
            else:
              p_i = math.floor(1 / (4 * (self.epsilon ** (f + (i - 1) * t))))
              if p_i < 1:
                grouped_items.extend((w, h, hc, hf) for w, h, hc, hf in items)
                continue

              items.sort(key=lambda x: x[WIDTH], reverse=True)
              for idx, (w, h, _, _) in enumerate(items):
                group_idx = idx // p_i
                first_in_group = group_idx * p_i
                bar_w = items[first_in_group][0]
                under_w = items[idx + p_i][0] if idx + p_i < r_j else 0
                grouped_items.append((bar_w, under_w, h, hc))

          # prepare wide item types: (width, height, count)
          wide_dict = Counter([(item[WIDTH], item[3]) for item in grouped_items])
          wide_types = [(w, hf, n) for (w, hf), n in wide_dict.items()]

          # prepare thin item types: (total_width, height)
          thin_dict = {}
          for w, h in thin:
            thin_dict[h] = thin_dict.get(h, 0) + w
          thin_types = [(w, h) for h, w in thin_dict.items()]

          # enumerate shelf configurations
          shelf_configs = self.enumerate_shelf_configs(wide_types)

          # (c.2.3) solve MIlP for this H
          try:
            profit = self.solve_shelf_mip(shelf_configs, wide_types, thin_types, total_height)
            pi[total_height] = max(pi.get(total_height, 0), profit)
          except RuntimeError:
            continue
      pi_list.append(pi)
    return pi_list

  def enumerate_shelf_configs(self, wide_types: List[Tuple[float, float, int]]) -> List[dict]:
    configs = []
    heights = set(h for _, h, _ in wide_types)
    for height in heights:
      for r in range(1, len(wide_types) + 1):
        for subset in combinations(wide_types, r):
          total_width = sum(w for w, h, _ in subset if h <= height)
          if total_width <= 1:
            wide_counts = {}
            for idx, (w, h, _) in enumerate(wide_types):
              count_in_subset = sum(1 for item in subset if item[0] == w and item[1] == h)
              if count_in_subset > 0:
                wide_counts[idx] = count_in_subset
            config = {
              'height': height,
              'wide_counts': wide_counts,
              'width_used': total_width,
              'profit': sum(w * h for w, h, _ in subset if h <= height)
            }
            configs.append(config)
    return configs

  def solve_shelf_mip(self, shelf_configs: List[dict], wide_types: List[Tuple[float, float, int]],
                      thin_types: List[Tuple[float, float]], H: float) -> float:
    prob = pulp.LpProblem("ShelfPacking", pulp.LpMaximize)

    # Variables
    x = {idx: pulp.LpVariable(f"x_{idx}", lowBound=0, cat="Integer") for idx in range(len(shelf_configs))}
    z = {j: pulp.LpVariable(f"z_{j}", lowBound=0, upBound=thin_types[j][0]) for j in range(len(thin_types))}

    # Objective: Maximize profit
    prob += (
            pulp.lpSum(shelf_configs[idx]['profit'] * x[idx] for idx in range(len(shelf_configs))) +
            pulp.lpSum(thin_types[j][1] * z[j] for j in range(len(thin_types)))
    )

    # Constraint: Total height limit
    prob += pulp.lpSum(shelf_configs[idx]['height'] * x[idx] for idx in range(len(shelf_configs))) <= H

    # Constraint: Wide item availability
    for l, (_, _, n_l) in enumerate(wide_types):
      prob += (
              pulp.lpSum(shelf_configs[idx]['wide_counts'].get(l, 0) * x[idx]
                         for idx in range(len(shelf_configs))) <= n_l
      )

    # Constraint: Thin item fitting
    for j, (w_j, h_j) in enumerate(thin_types):
      prob += (
              pulp.lpSum((1 - shelf_configs[idx]['width_used']) * x[idx]
                         for idx in range(len(shelf_configs))
                         if shelf_configs[idx]['height'] >= h_j) >= z[j]
      )

    # Solve the MIP
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[prob.status] != 'Optimal':
      raise RuntimeError("mip could not be solved optimally")
    return pulp.value(prob.objective)

  # part d
  def high_shelves(self, A0_shelf_configs, A0_thin_items, pi_list):
    """
    Processes multi-shelf configurations, packs thin items, and uses remaining height for lower shelves.
    """
    best_profit = 0
    best_config = None

    for config in A0_shelf_configs:
      # prepare shelves for thin item packing
      shelves = config['shelves']
      for shelf in shelves:
        shelf['width_left'] = 1.0 - shelf['width_used']

      # print(shelves)
      # pack thin items into the shelves
      thin_area = fill_thin([shelf.copy() for shelf in shelves], A0_thin_items.copy())

      # Total profit from wide items in the configuration
      wide_profit = config['total_profit']

      # Total height used by the tall shelves
      used_height = config['total_height']
      residual = 1.0 - used_height

      # Use remaining height for lower shelves via multiple-choice knapsack
      mc_profit = multiple_choice_knapsack(pi_list, residual)

      # Total profit for this configuration
      total = wide_profit + thin_area + mc_profit

      if total > best_profit:
        best_profit = total
        best_config = config

    return best_profit, best_config

    # part e

  def fill_thin_with_cuts(self, shelves, thin_items):
    """
    Wrapper for existing fill_thin to track cut items.
    Returns total_area and list of cut items.
    """
    total_area = 0
    cut_items = []
    open_shelves = shelves[:]
    thin = thin_items[:]
    idx_thin = 0

    # Simulate the existing fill_thin logic
    while open_shelves and idx_thin < len(thin):
      w, h = thin[idx_thin]
      shelf = open_shelves[0]
      if w <= shelf['width_left']:
        total_area += w * h
        shelf['width_left'] -= w
        idx_thin += 1
      else:
        # Item is cut vertically
        frac_w = shelf['width_left']
        total_area += frac_w * h
        cut_items.append((w, h))  # Record original item that was cut
        open_shelves.pop(0)  # Shelf is full
      return total_area, cut_items

  def solve(self):
    f, A_groups = self.decomposition()
    # print(f, A_groups)
    if not f:
      print("no f")
      return {'profit': 0.0}
    wide_groups, thin_items_groups = self.subdivide_and_melt(f, A_groups)
    # print(wide_groups, thin_items_groups)
    pi_list = self.low_shelves(wide_groups, thin_items_groups, f)
    A0_shelf_configs = self.enumerate_A0_shelves(wide_groups[0])
    result = self.high_shelves(A0_shelf_configs, thin_items_groups, pi_list)
    return result

  def enumerate_A0_shelves(self, wide_items_A0):
    """
    Enumerates all possible multi-shelf configurations for wide items in A_0.
    Each configuration is a list of shelves where total height <= 1 and shelves are disjoint.
    """
    # Step 1: Generate all possible single-shelf configurations
    single_shelves = []
    for r in range(1, len(wide_items_A0) + 1):
      for subset in itertools.combinations(wide_items_A0, r):
        total_width = sum(item[0] for item in subset)
        if total_width <= 1:
          shelf_height = max(item[1] for item in subset)
          profit = sum(item[0] * item[1] for item in subset)
          items = frozenset(subset)  # Use frozenset for efficient set operations
          single_shelves.append({
            'height': shelf_height,
            'profit': profit,
            'items': items,
            'width_used': total_width
          })

    # Step 2: Generate multi-shelf configurations recursively
    def generate_configs(available_shelves, used_items, current_height, current_profit, current_shelves):
      if current_height > 1:
        return
      # Yield the current configuration (could be empty or have multiple shelves)
      yield {
        'shelves': current_shelves[:],
        'total_height': current_height,
        'total_profit': current_profit
      }
      # Try adding each remaining shelf
      for i, shelf in enumerate(available_shelves):
        if shelf['items'].isdisjoint(used_items):
          new_used_items = used_items.union(shelf['items'])
          new_height = current_height + shelf['height']
          new_profit = current_profit + shelf['profit']
          new_shelves = current_shelves + [shelf]
          # Recurse with remaining shelves
          yield from generate_configs(available_shelves[i + 1:], new_used_items, new_height, new_profit, new_shelves)

    # Generate all configurations
    configurations = list(generate_configs(single_shelves, frozenset(), 0, 0, []))
    return configurations


# utility functions
def fill_thin(shelves, thin_items):
  open_shelves = sorted(shelves, key=lambda s: -s['height'])
  # print(thin_items, shelves)
  flat_thin = [item for group in thin_items for item in group]
  thin = sorted(flat_thin, key=lambda x: -x[1])
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
