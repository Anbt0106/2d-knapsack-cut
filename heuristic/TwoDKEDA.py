import random
import numpy as np
from copy import deepcopy
from typing import List, Tuple

# --- Data structures ---
class Rectangle:
    def __init__(self, width: float, height: float):
        # Normalize to [0,1]
        self.w = width
        self.h = height
        self.area = width * height

class Individual:
    def __init__(self):
        self.layers: List[List[Rectangle]] = []
        self.fitness: float = 0.0

# --- 2D-KEDA Solver ---
class TwoDKEDA:
    def __init__(self,
                 items: List[Tuple[float, float]],
                 max_time: float = 60.0,
                 pop_size: int = 20,
                 keep_frac: float = 0.2,
                 alpha: float = 0.3):
        # Problem: pack in 1x1 strip using guillotine cuts
        self.items = [Rectangle(w, h) for w, h in items]
        self.N = len(self.items)
        self.max_time = max_time
        # EDA parameters
        self.pop_size = pop_size
        self.keep = max(1, int(pop_size * keep_frac))
        self.alpha = alpha
        # Probability models
        self.p_item = np.ones(self.N) / self.N
        self.p_pair = np.ones((self.N, self.N)) / self.N

    def evaluate(self, ind: Individual):
        # Simple bottom-up DP 2D guillotine packing on each layer
        # Here each layer has height = max h in layer,
        # and width <=1 by construction from placement
        fitness = 0.0
        for layer in ind.layers:
            # pack by next-fit stack: check if width sum <=1
            w_sum = sum(r.w for r in layer)
            h_max = max((r.h for r in layer), default=0)
            if w_sum <= 1.0:
                fitness += w_sum * h_max
        ind.fitness = fitness

    def initial_population(self) -> List[Individual]:
        pop = []
        for _ in range(self.pop_size):
            ind = Individual()
            remaining = self.items[:]
            while remaining:
                # sample subset Q by p_item
                Q = [r for i, r in enumerate(remaining)
                     if random.random() < self.p_item[i]]
                if not Q:
                    break
                # form a layer greedily
                layer = sorted(Q, key=lambda rx: rx.area, reverse=True)
                w_sum = 0.0
                packed = []
                for rect in layer:
                    if w_sum + rect.w <= 1.0:
                        packed.append(rect)
                        w_sum += rect.w
                ind.layers.append(packed)
                # remove packed
                for r in packed:
                    remaining.remove(r)
            self.evaluate(ind)
            pop.append(ind)
        return pop

    def update_models(self, pop: List[Individual]):
        # select top keep
        best = sorted(pop, key=lambda x: x.fitness, reverse=True)[:self.keep]
        # update p_item
        freq = np.zeros(self.N)
        for ind in best:
            for layer in ind.layers:
                for r in layer:
                    idx = self.items.index(r)
                    freq[idx] += 1
        freq /= (len(best) * sum(len(layer) for ind in best for layer in ind.layers))
        self.p_item = (1 - self.alpha) * self.p_item + self.alpha * freq

    def evolve(self):
        import time
        start = time.time()
        pop = self.initial_population()
        best = max(pop, key=lambda x: x.fitness)
        while time.time() - start < self.max_time:
            # generate offspring
            offspring = []
            for _ in range(self.pop_size):
                # sample new individual
                child = Individual()
                remaining = self.items[:]
                while remaining:
                    Q = [r for i, r in enumerate(remaining)
                         if random.random() < self.p_item[i]]
                    if not Q:
                        break
                    # greedy pack
                    layer = []
                    w_sum = 0.0
                    for rect in sorted(Q, key=lambda rx: rx.area, reverse=True):
                        if w_sum + rect.w <= 1.0:
                            layer.append(rect)
                            w_sum += rect.w
                    child.layers.append(layer)
                    for r in layer:
                        remaining.remove(r)
                self.evaluate(child)
                offspring.append(child)
            # combine and select
            pop.extend(offspring)
            pop = sorted(pop, key=lambda x: x.fitness, reverse=True)[:self.pop_size]
            # update
            self.update_models(pop)
            # track best
            if pop[0].fitness > best.fitness:
                best = deepcopy(pop[0])
        return {'profit': best.fitness, 'layers': best.layers}

