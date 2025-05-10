import random
import numpy as np
from copy import deepcopy
import time
import tracemalloc
from typing import List, Tuple
import uuid


# Rectangle class to store properties
class Rectangle:
    def __init__(self, index: int, width: float, height: float, profit: float):
        self.index = index
        self.id = str(uuid.uuid4())
        self.width = width
        self.height = height
        self.profit = profit


# Individual class to store a solution
class Individual:
    def __init__(self):
        self.layers = []
        self.rem = []
        self.fitness = 0
        self.total_height = 0
        self.cutting_commands = []


class TwoDKEDA:
    def __init__(self, items: List[Tuple[float, float]], max_time: float = 60):
        self.W, self.H = 100, 100  # Strip dimensions
        self.t, self.tmax = 3, 10  # Population size
        self.kn = 5  # Update probability models every kn generations
        self.timeend = max_time  # Time limit
        self.gp, self.rp = 300, 0.7  # Restart parameters
        self.tp = 0.1  # Truncation selection percentage
        self.p_block = 0.5  # Probability for initial block building
        self.p_of = 0.5  # Probability of choosing HP1 over HP2
        self.p_imp = 0.25  # Probability of using ImpLS
        self.p_swap = 0.5  # Probability of using sampling vs. selection+mutation
        self.LSremn = 10  # Number of local search iterations
        self.alpha = 0.2  # Relaxation factor for probability models

        # Convert items to Rectangle objects
        self.rectangles = [
            Rectangle(i, width, height, width * height)  # Profit as area
            for i, (width, height) in enumerate(items)
        ]
        self.n = len(self.rectangles)
        # Initialize probability models
        self.M1 = np.ones(self.n) * 0.5
        self.ECM = np.ones((self.n, self.n)) * 0.5

    def check_fit(self, rect: Rectangle, vbw: float, vbh: float) -> bool:
        return rect.width <= vbw and rect.height <= vbh

    def find_fit_list(self, Q: List[Rectangle], vbw: float, vbh: float) -> List[Rectangle]:
        fit_list = [rect for rect in Q if self.check_fit(rect, vbw, vbh)]
        return fit_list[:min(len(fit_list), 200)]

    def HP1(self, Q: List[Rectangle], vbw: float, vbh: float) -> Rectangle:
        fit_list = self.find_fit_list(Q, vbw, vbh)
        return max(fit_list, key=lambda r: (r.profit, r.width)) if fit_list else None

    def HP2(self, Q: List[Rectangle], vbw: float, vbh: float) -> Rectangle:
        fit_list = self.find_fit_list(Q, vbw, vbh)
        return random.choice(fit_list) if fit_list else None

    def ImpLS(self, Q: List[Rectangle], vbw: float, vbh: float, fit_list: List[Rectangle]) -> Tuple[
        List[Rectangle], float]:
        best_block, best_profit, max_height = None, 0, 0
        max_combinations, count = 100, 0
        for i in range(len(fit_list)):
            block = [fit_list[i]]
            width, height, profit = fit_list[i].width, fit_list[i].height, fit_list[i].profit
            if width <= vbw and height <= vbh and profit > best_profit:
                best_profit, best_block, max_height = profit, block, height
            if count >= max_combinations:
                break
            for j in range(i + 1, len(fit_list)):
                new_width = width + fit_list[j].width
                if new_width > vbw:
                    continue
                block2 = block + [fit_list[j]]
                height2 = max(height, fit_list[j].height)
                profit2 = profit + fit_list[j].profit
                if height2 <= vbh and profit2 > best_profit:
                    best_profit, best_block, max_height = profit2, block2, height2
                if count >= max_combinations:
                    break
                for k in range(j + 1, len(fit_list)):
                    new_width2 = new_width + fit_list[k].width
                    if new_width2 > vbw:
                        continue
                    block3 = block2 + [fit_list[k]]
                    height3 = max(height2, fit_list[k].height)
                    profit3 = profit2 + fit_list[k].profit
                    if height3 <= vbh and profit3 > best_profit:
                        best_profit, best_block, max_height = profit3, block3, height3
                    count += 1
                    if count >= max_combinations:
                        break
        return best_block, max_height if best_block else (None, 0)

    def placement_strategy(self, Q: List[Rectangle], vbw: float, vbh: float, x00: float, y00: float) -> Tuple[
        float, float, List[Tuple[Rectangle, float, float]]]:
        fit_list = self.find_fit_list(Q, vbw, vbh)
        if not fit_list:
            return 0, 0, []
        rect = self.HP1(Q, vbw, vbh) if random.random() < self.p_of else self.HP2(Q, vbw, vbh)
        if not rect:
            return 0, 0, []
        plw, plh = rect.width, rect.height
        placed = [(rect, x00, y00)]
        if random.random() < self.p_imp:
            block, max_height = self.ImpLS(Q, vbw - plw, vbh, fit_list)
            if block:
                plw = sum(r.width for r in block)
                plh = max_height
                placed = [(r, x00 + sum(r2.width for r2 in block[:i]), y00) for i, r in enumerate(block)]
        return plw, plh, placed

    def packlayer(self, vbw: float, vbh: float, x00: float, y00: float, Q: List[Rectangle], individual: Individual):
        if vbh <= 0 or vbw <= 0 or not Q:
            return
        plw, plh, placed = self.placement_strategy(Q, vbw, vbh, x00, y00)
        if not placed:
            return
        for rect, x, y in placed:
            individual.layers[-1].append((rect, x, y))
            individual.fitness += rect.profit
            Q.remove(rect)
        dw = vbw - plw
        min_width = min((r.width for r in Q), default=vbw) if Q else vbw
        if dw < min_width:
            individual.cutting_commands.append(f"H({x00},{y00 + plh})")
            self.packlayer(vbw, vbh - plh, x00, y00 + plh, Q, individual)
        else:
            if plw * (vbh - plh) <= (vbw - plw) * vbh:
                individual.cutting_commands.append(f"H({x00},{y00 + plh})")
                m1 = vbw
                self.packlayer(plw, vbh - plh, x00, y00 + plh, Q, individual)
                individual.cutting_commands.append(f"V({x00 + plw},{y00})")
                self.packlayer(m1 - plw, plh, x00 + plw, y00, Q, individual)
            else:
                individual.cutting_commands.append(f"V({x00 + plw},{y00})")
                self.packlayer(vbw - plw, vbh, x00 + plw, y00, Q, individual)
                individual.cutting_commands.append(f"H({x00},{y00 + plh})")
                self.packlayer(plw, vbh - plh, x00, y00 + plh, Q, individual)

    def repacking(self, layer: List[Tuple[Rectangle, float, float]], target_height: float) -> Tuple[
        List[Tuple[Rectangle, float, float]], List[str]]:
        Q = [rect for rect, _, _ in layer]
        new_individual = Individual()
        new_individual.layers.append([])
        self.packlayer(self.W, target_height, 0, 0, Q, new_individual)
        if not new_individual.layers[0] or Q:
            return [], []
        return new_individual.layers[0], new_individual.cutting_commands

    def repair(self, individual: Individual):
        max_attempts = 100
        attempts = 0
        while individual.total_height > self.H and individual.layers and attempts < max_attempts:
            layer_idx = random.randint(0, len(individual.layers) - 1)
            if individual.layers[layer_idx]:
                rect, _, _ = random.choice(individual.layers[layer_idx])
                individual.layers[layer_idx].remove((rect, _, _))
                individual.rem.append(rect)
                individual.fitness -= rect.profit
                layer_height = max((r.height for r, _, _ in individual.layers[layer_idx]), default=0)
                new_layer, new_commands = self.repacking(individual.layers[layer_idx], layer_height)
                if new_layer:
                    individual.layers[layer_idx] = new_layer
                    individual.cutting_commands = [cmd for cmd in individual.cutting_commands if
                                                   not cmd.startswith(f"H(0,{layer_idx * layer_height})")]
                    individual.cutting_commands.extend(new_commands)
                else:
                    individual.fitness -= sum(r.profit for r, _, _ in individual.layers[layer_idx])
                    individual.rem.extend(r for r, _, _ in individual.layers[layer_idx])
                    individual.layers.pop(layer_idx)
                individual.total_height = sum(
                    max((r.height for r, _, _ in layer), default=0) for layer in individual.layers)
            attempts += 1

    def LSrem1(self, individual: Individual) -> bool:
        if not individual.rem:
            return False
        rect = random.choice(individual.rem)
        layer_idx = random.randint(0, len(individual.layers) - 1) if individual.layers else 0
        if not individual.layers:
            individual.layers.append([])
        layer = individual.layers[layer_idx]
        layer_height = max((r.height for r, _, _ in layer), default=0)
        new_layer = layer + [(rect, 0, 0)]
        new_layer, new_commands = self.repacking(new_layer, layer_height)
        if new_layer and individual.total_height - layer_height + max((r.height for r, _, _ in new_layer),
                                                                      default=0) <= self.H:
            individual.layers[layer_idx] = new_layer
            individual.rem.remove(rect)
            individual.fitness += rect.profit
            individual.cutting_commands = [cmd for cmd in individual.cutting_commands if
                                           not cmd.startswith(f"H(0,{layer_idx * layer_height})")]
            individual.cutting_commands.extend(new_commands)
            individual.total_height = sum(
                max((r.height for r, _, _ in layer), default=0) for layer in individual.layers)
            return True
        return False

    def LSrem2(self, individual: Individual) -> bool:
        if not individual.rem:
            return False
        group_size = min(random.randint(2, 3), len(individual.rem))
        group = random.sample(individual.rem, group_size)
        layer_idx = random.randint(0, len(individual.layers) - 1) if individual.layers else 0
        if not individual.layers:
            individual.layers.append([])
        layer = individual.layers[layer_idx]
        layer_height = max((r.height for r, _, _ in layer), default=0)
        new_layer = layer + [(r, 0, 0) for r in group]
        new_layer, new_commands = self.repacking(new_layer, layer_height)
        if new_layer and individual.total_height - layer_height + max((r.height for r, _, _ in new_layer),
                                                                      default=0) <= self.H:
            individual.layers[layer_idx] = new_layer
            for r in group:
                individual.rem.remove(r)
                individual.fitness += r.profit
            individual.cutting_commands = [cmd for cmd in individual.cutting_commands if
                                           not cmd.startswith(f"H(0,{layer_idx * layer_height})")]
            individual.cutting_commands.extend(new_commands)
            individual.total_height = sum(
                max((r.height for r, _, _ in layer), default=0) for layer in individual.layers)
            return True
        return False

    def LSrem3(self, individual: Individual) -> bool:
        if not individual.rem:
            return False
        layer_idx = random.randint(0, len(individual.layers) - 1) if individual.layers else 0
        if not individual.layers:
            individual.layers.append([])
        layer = individual.layers[layer_idx]
        layer_height = max((r.height for r, _, _ in layer), default=0)
        new_layer = layer + [(r, 0, 0) for r in individual.rem]
        new_layer, new_commands = self.repacking(new_layer, layer_height)
        if new_layer and individual.total_height - layer_height + max((r.height for r, _, _ in new_layer),
                                                                      default=0) <= self.H:
            individual.layers[layer_idx] = new_layer
            individual.fitness += sum(r.profit for r in individual.rem)
            individual.rem = []
            individual.cutting_commands = [cmd for cmd in individual.cutting_commands if
                                           not cmd.startswith(f"H(0,{layer_idx * layer_height})")]
            individual.cutting_commands.extend(new_commands)
            individual.total_height = sum(
                max((r.height for r, _, _ in layer), default=0) for layer in individual.layers)
            return True
        return False

    def sample_M1(self) -> Tuple[List[Rectangle], List[Rectangle]]:
        QK = [rect for i, rect in enumerate(self.rectangles) if random.random() < self.M1[i]]
        rem = [r for r in self.rectangles if r not in QK]
        return QK, rem

    def sample_ECM(self, QK: List[Rectangle], individual: Individual):
        while QK:
            Q = []
            rect_i = random.choice(QK)
            Q.append(rect_i)
            QK.remove(rect_i)
            pr = self.ECM[rect_i.index] / self.ECM[rect_i.index].sum() if self.ECM[rect_i.index].sum() > 0 else np.ones(
                self.n) / self.n
            for rect_j in QK[:]:
                if random.random() < pr[rect_j.index]:
                    Q.append(rect_j)
                    QK.remove(rect_j)
            individual.layers.append([])
            self.packlayer(self.W, self.H - individual.total_height, 0, individual.total_height, Q, individual)
            if individual.layers[-1]:
                layer_height = max((r.height for r, _, _ in individual.layers[-1]), default=0)
                individual.total_height += layer_height
                individual.cutting_commands.append(f"H(0,{individual.total_height})")
                new_layer, new_commands = self.repacking(individual.layers[-1], layer_height * 0.9)
                if new_layer:
                    individual.layers[-1] = new_layer
                    individual.cutting_commands = [cmd for cmd in individual.cutting_commands if
                                                   not cmd.endswith(f",{individual.total_height})")]
                    individual.cutting_commands.extend(new_commands)
                    individual.total_height = sum(
                        max((r.height for r, _, _ in layer), default=0) for layer in individual.layers)
            for rect in QK + individual.rem:
                temp_layer = individual.layers[-1] + [(rect, 0, 0)]
                new_layer, new_commands = self.repacking(temp_layer, layer_height)
                if new_layer and sum(r.profit for r, _, _ in new_layer) > sum(
                        r.profit for r, _, _ in individual.layers[-1]):
                    individual.layers[-1] = new_layer
                    individual.fitness = sum(r.profit for r, _, _ in new_layer) - sum(
                        r.profit for r, _, _ in individual.layers[-1] if r != rect)
                    individual.fitness += rect.profit
                    if rect in QK:
                        QK.remove(rect)
                    if rect in individual.rem:
                        individual.rem.remove(rect)
            if individual.total_height > self.H:
                self.repair(individual)

    def mutation(self, individual: Individual, best_individual: Individual) -> Individual:
        if len(individual.layers) < 2 or any(len(layer) < 3 for layer in individual.layers):
            return individual
        layer1_idx = random.randint(0, len(individual.layers) - 1)
        layer1 = individual.layers[layer1_idx]
        if len(layer1) < 3:
            return individual
        rect_i, rect_j = random.sample(layer1, 2)
        rect_k = random.choice([r for r, _, _ in layer1 if r not in [rect_i[0], rect_j[0]]])
        for layer in best_individual.layers:
            rect_i_in_layer = any(r.index == rect_i[0].index for r, _, _ in layer)
            rect_k_in_layer = any(r.index == rect_k.index for r, _, _ in layer)
            if rect_i_in_layer and rect_k_in_layer:
                return individual
        other_layers = [l for i, l in enumerate(individual.layers) if i != layer1_idx]
        if not other_layers:
            return individual
        layer2 = random.choice(other_layers)
        if not layer2:
            return individual
        rect_z = max(layer2, key=lambda x: self.ECM[rect_i[0].index][x[0].index])[0]
        for item in layer1[:]:
            if item[0].index == rect_k.index:
                layer1.remove(item)
                break
        for item in layer2[:]:
            if item[0].index == rect_z.index:
                layer2.remove(item)
                break
        layer1.append((rect_z, 0, 0))
        layer2.append((rect_k, 0, 0))
        layer1_height = max((r.height for r, _, _ in layer1), default=0)
        layer2_height = max((r.height for r, _, _ in layer2), default=0)
        new_layer1, new_commands1 = self.repacking(layer1, layer1_height)
        new_layer2, new_commands2 = self.repacking(layer2, layer2_height)
        if new_layer1 and new_layer2:
            individual.layers[layer1_idx] = new_layer1
            individual.layers[individual.layers.index(layer2)] = new_layer2
            individual.fitness = sum(r.profit for layer in individual.layers for r, _, _ in layer)
            individual.cutting_commands = [cmd for cmd in individual.cutting_commands if not (
                        cmd.startswith(f"H(0,{layer1_idx * layer1_height})") or cmd.startswith(
                    f"H(0,{individual.layers.index(layer2) * layer2_height})"))]
            individual.cutting_commands.extend(new_commands1 + new_commands2)
            individual.total_height = sum(
                max((r.height for r, _, _ in layer), default=0) for layer in individual.layers)
            if individual.total_height > self.H:
                self.repair(individual)
        return individual

    def update_probability_models(self, population: List[Individual]):
        best_count = max(1, int(0.2 * len(population)))
        best_individuals = sorted(population, key=lambda ind: ind.fitness, reverse=True)[:best_count]
        delta_M1 = np.zeros(self.n)
        for ind in best_individuals:
            for layer in ind.layers:
                for rect, _, _ in layer:
                    delta_M1[rect.index] += 1
        delta_M1 /= best_count
        self.M1 = (1 - self.alpha) * self.M1 + self.alpha * delta_M1
        delta_ECM = np.zeros((self.n, self.n))
        for ind in best_individuals:
            for layer in ind.layers:
                rect_indices = [r.index for r, _, _ in layer]
                for i in rect_indices:
                    for j in rect_indices:
                        if i != j:
                            delta_ECM[i][j] += 1
        delta_ECM /= best_count
        self.ECM = (1 - self.alpha) * self.ECM + self.alpha * delta_ECM

    def generate_initial_population(self) -> List[Individual]:
        population = []
        for _ in range(self.t):
            individual = Individual()
            Q = self.rectangles[:]
            while Q and individual.total_height < self.H:
                layer_height = random.uniform(0.3 * self.H, self.H - individual.total_height)
                individual.layers.append([])
                Q_copy = Q[:]
                self.packlayer(self.W, layer_height, 0, individual.total_height, Q_copy, individual)
                if individual.layers[-1]:
                    individual.total_height += max((r.height for r, _, _ in individual.layers[-1]), default=0)
                    individual.cutting_commands.append(f"H(0,{individual.total_height})")
                    Q = Q_copy
                else:
                    individual.layers.pop()
                individual.rem = Q
                self.repair(individual)
            population.append(individual)
        return population

    def restart(self, population: List[Individual]) -> List[Individual]:
        if len(population) <= self.t:
            return population
        population.sort(key=lambda ind: ind.fitness, reverse=True)
        keep_count = max(self.t, int((1 - self.rp) * len(population)))
        return population[:keep_count]

    def reinsertion(self, parent: Individual, descendant: Individual, population: List[Individual]) -> List[Individual]:
        if descendant.fitness > parent.fitness:
            population[population.index(parent)] = descendant
        elif len(population) < self.tmax:
            population.append(descendant)
        return population

    def solve(self) -> dict:
        start_time = time.time()
        population = self.generate_initial_population()
        best_fitness = 0
        stagnant_generations = 0
        generation = 0
        while time.time() - start_time < self.timeend:
            for _ in range(self.kn):
                descendant = Individual()
                if random.random() < self.p_swap:
                    QK, descendant.rem = self.sample_M1()
                    self.sample_ECM(QK, descendant)
                else:
                    best_count = max(1, int(self.tp * len(population)))
                    parent = random.choice(sorted(population, key=lambda ind: ind.fitness, reverse=True)[:best_count])
                    descendant = deepcopy(parent)
                    best_individual = max(population, key=lambda ind: ind.fitness)
                    for _ in range(3):
                        descendant = self.mutation(descendant, best_individual)
                for _ in range(self.LSremn):
                    self.LSrem1(descendant)
                    self.LSrem2(descendant)
                    self.LSrem3(descendant)
                parent = random.choice(population)
                population = self.reinsertion(parent, descendant, population)
            if len(population) < self.tmax:
                population.append(descendant)
            for _ in range(self.LSremn):
                self.LSrem1(descendant)
                self.LSrem2(descendant)
                self.LSrem3(descendant)
            best_individual = max(population, key=lambda ind: ind.fitness)
            for _ in range(self.LSremn):
                self.LSrem1(best_individual)
                self.LSrem2(best_individual)
                self.LSrem3(best_individual)
            if generation % 3 == 0:
                self.update_probability_models(population)
            current_best = max(population, key=lambda ind: ind.fitness).fitness
            if current_best <= best_fitness:
                stagnant_generations += 1
            else:
                best_fitness = current_best
                stagnant_generations = 0
            if stagnant_generations >= self.gp:
                population = self.restart(population)
                stagnant_generations = 0
            generation += 1
        best_individual = max(population, key=lambda ind: ind.fitness)
        return {"profit": best_individual.fitness}
