from APTAS import *

class PTAS:

  def __init__(self, items: List[Tuple[float, float]], epsilon: float):
    """
    Initialize the PTAS class with items and epsilon.

    Args:
        items: List of tuples (width, height) representing items.
        epsilon: Accuracy parameter (0 < epsilon <= 1).
    """
    self.items = items
    self.epsilon = epsilon
    self.T = [item for item in items if item[1] >= epsilon]  # Tall items
    self.W = [item for item in items if item[0] >= epsilon ** 4]  # Wide items
    self.thin_items = [item for item in self.T if item[0] < epsilon ** 4]  # Thin tall items


def nfdh_pack(self, items: List[Tuple[float, float]], bin_height: float = 1.0) -> bool:
  """
  Check if Next-Fit Decreasing Height (NFDH) can pack all items within the given bin height.

  Args:
      items: List of items to pack.
      bin_height: Maximum height of the bin (default 1.0).

  Returns:
      bool: True if all items can be packed, False otherwise.
  """
  sorted_items = sorted(items, key=lambda x: -x[1])  # Sort by decreasing height
  current_height = 0.0
  shelf_width = 0.0
  shelf_height = 0.0
  for item in sorted_items:
    if shelf_width + item[0] > 1.0:
      current_height += shelf_height
      if current_height + item[1] > bin_height:
        return False
      shelf_width = item[0]
      shelf_height = item[1]
    else:
      shelf_width += item[0]
      shelf_height = max(shelf_height, item[1])
  current_height += shelf_height
  return current_height <= bin_height


def knapsack_ptas(self, items: List[Tuple[float, float]], capacity: float, epsilon: float) -> float:
  """
  1D Knapsack PTAS for thin items with width as weight and w*h as profit.

  Args:
      items: List of thin items.
      capacity: Available width capacity.
      epsilon: Accuracy parameter.

  Returns:
      float: Approximate maximum profit.
  """
  if not items or all(w == 0 for w, h in items):
    return sum(w * h for w, h in items)
  n = len(items)
  P = max(w * h for w, h in items)  # Max profit
  if P == 0:
    return 0.0
  k = math.ceil(1 / epsilon)
  sorted_items = sorted(items, key=lambda x: -(x[0] * x[1]))
  if n <= k:
    total_width = sum(item[0] for item in sorted_items)
  if total_width <= capacity:
    return sum(w * h for w, h in sorted_items)
  top_k = sorted_items[:k]
  top_width = sum(item[0] for item in top_k)
  top_profit = sum(w * h for w, h in top_k)
  if top_width <= capacity:
    return top_profit
  remaining = sorted_items[k:]
  min_width = min(w for w, h in remaining if w > 0)
  scale = epsilon * P / n
  scaled_profits = [int((w * h) / scale) for w, h in remaining]
  W = int(capacity / (epsilon * min_width))
  dp = [0] * (W + 1)
  for i, (w, h) in enumerate(remaining):
    w_scaled = int(w / (epsilon * min_width))
  p_scaled = scaled_profits[i]
  for j in range(W, w_scaled - 1, -1):
    if dp[j - w_scaled] + p_scaled > dp[j]:
      dp[j] = dp[j - w_scaled] + p_scaled
  max_w = min(W, int(capacity / min_width))
  profit = dp[max_w] * scale
  return profit


def pack_tall_shelf(self, S: List[Tuple[float, float]], H: float) -> float:
  """
  Pack tall shelf of height H with given items (wide and thin).

  Args:
      S: List of tall items for the shelf.
      H: Height of the tall shelf.

  Returns:
      float: Total profit from the tall shelf.
  """
  S_wide = [item for item in S if item[0] >= self.epsilon ** 4]
  w_S = sum(item[0] for item in S_wide)
  remaining_width = 1.0 - w_S
  thin_candidates = [item for item in self.thin_items if item[1] <= H and item not in S]
  if w_S >= 1 - self.epsilon ** 3:
    thin_profit = self.knapsack_ptas(thin_candidates, remaining_width, self.epsilon)
  else:
  sorted_thin = sorted(thin_candidates, key=lambda x: -x[1])
  packed_thin = []
  current_width = 0.0
  for item in sorted_thin:
    if current_width + item[0] <= remaining_width:
      packed_thin.append(item)
    current_width += item[0]
  thin_profit = sum(w * h for w, h in packed_thin)
  tall_profit = sum(w * h for w, h in S) + thin_profit
  return tall_profit


def solve(self) -> float:
  """
  Solve the 2SKP using PTAS, integrating the APTAS class.

  Returns:
      float: Approximate maximum profit.
  """
  N_minus_T = [item for item in self.items if item not in self.T]  # non-tall
  # case 1: not all non-tall items can be packed using nfdh
  if not self.nfdh_pack(N_minus_T):
    aptas = APTAS(self.items, self.epsilon)
  return aptas.solve()[0]

  # case 2: total width of tall items is large (> epsilon^3)
  w_T = sum(item[0] for item in self.T)  # total width
  if w_T >= self.epsilon ** 3:
    accuracy = self.epsilon ** 2 * min(1, w_T)
  aptas = APTAS(self.items, accuracy)
  return aptas.solve()[0]

  # case 3: all non-tall items can be packed and w(T) < epsilon^3
  profit_N_minus_T = sum(w * h for w, h in N_minus_T)
  best_profit = profit_N_minus_T

  # Since w(T) < epsilon^3 and w_j >= epsilon^4 for tall items, |T| <= 1/epsilon -
  # at most 1 / epsilon tall items
  for r in range(1, len(self.T) + 1):
    for S_tuple in combinations(self.T, r):
      S = list(S_tuple)
      if sum(item[0] for item in S) > 1:
        continue

      H = max(item[HEIGHT] for item in S)
      tall_profit = self.pack_tall_shelf(S, H)  # pack item with height H
      N_prime = [item for item in self.items if item not in S]
      T_prime = [item for item in N_prime if item[HEIGHT] >= self.epsilon]
      N_prime_minus_T_prime = [item for item in N_prime if item not in T_prime]  # N' \ T'

      if not self.nfdh_pack(N_prime_minus_T_prime, 1 - H):  # try packing the remaining height
        aptas = APTAS(N_prime, self.epsilon)
      lower_profit = aptas.solve()[0]
      else:
      w_T_prime = sum(item[WIDTH] for item in T_prime)
      if w_T_prime >= self.epsilon ** 3:
        accuracy = self.epsilon ** 2 * min(1, w_T_prime)
        aptas = APTAS(N_prime, accuracy)
        lower_profit = aptas.solve()[0]
      else:
        lower_profit = sum(w * h for w, h in N_prime_minus_T_prime)
      total_profit = tall_profit + lower_profit
      best_profit = max(best_profit, total_profit)

  return best_profit
