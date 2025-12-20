from datetime import datetime
from typing import List

from .population import Population
from .ga_core import GACore

class NSGAII:
    def __init__(self, core: GACore):
        self.core = core

    # ... (Giữ nguyên các hàm helper _calculate_dominants, _calculate_fronts, v.v...) ...
    def _calculate_dominants(self, pop: Population):
        for i in range(len(pop.population)):
            pop.dominated_by[i] = set()
            pop.dominates_to[i] = set()
            pop.fronts[i] = set()

        def dominates(a, b):
            fa = pop.fitness[a]
            fb = pop.fitness[b]
            better_or_equal = (
                fa.get("latency", 0) <= fb.get("latency", 0) and
                fa.get("spread", 0) <= fb.get("spread", 0) and
                fa.get("underutilization", 0) <= fb.get("underutilization", 0)
            )
            strictly_better = (
                fa.get("latency", 0) < fb.get("latency", 0) or
                fa.get("spread", 0) < fb.get("spread", 0) or
                fa.get("underutilization", 0) < fb.get("underutilization", 0)
            )
            return better_or_equal and strictly_better

        for p in range(len(pop.population)):
            for q in range(p + 1, len(pop.population)):
                if dominates(p, q):
                    pop.dominates_to[p].add(q)
                    pop.dominated_by[q].add(p)
                elif dominates(q, p):
                    pop.dominates_to[q].add(p)
                    pop.dominated_by[p].add(q)

    def _calculate_fronts(self, pop: Population):
        added = set()
        i = 0
        while len(added) < len(pop.population):
            pop.fronts[i] = set([idx for idx, dom in enumerate(pop.dominated_by) if dom == set() and idx not in added])
            added = added | pop.fronts[i]
            for idx in list(pop.fronts[i]):
                for j in pop.dominates_to[idx]:
                    if idx in pop.dominated_by[j]:
                        pop.dominated_by[j].remove(idx)
            i += 1

    def _crowding_distances_assign(self, pop: Population, front: List[int]):
        for i in front:
            pop.crowding_distances[i] = 0.0

        if not front:
            return
        front_fitness = [pop.fitness[i] for i in front]

        for key in ("latency", "spread", "underutilization"):
            sorted_idx = sorted(front, key=lambda k: pop.fitness[k].get(key, 0))
            if len(sorted_idx) > 0:
                pop.crowding_distances[sorted_idx[0]] = float("inf")
                if len(sorted_idx) > 1:
                    pop.crowding_distances[sorted_idx[-1]] = float("inf")
                    fmin = pop.fitness[sorted_idx[0]].get(key, 0)
                    fmax = pop.fitness[sorted_idx[-1]].get(key, 0)
                    denom = (fmax - fmin) if (fmax - fmin) != 0 else 1.0
                    for k in range(1, len(sorted_idx) - 1):
                        prev = pop.fitness[sorted_idx[k - 1]].get(key, 0)
                        nxt = pop.fitness[sorted_idx[k + 1]].get(key, 0)
                        pop.crowding_distances[sorted_idx[k]] += (nxt - prev) / denom

    def _calculate_crowding(self, pop: Population):
        i = 0
        while len(pop.fronts[i]) != 0:
            self._crowding_distances_assign(pop, list(pop.fronts[i]))
            i += 1

    def _crowded_comparison_order(self, pop: Population):
        values = []
        rank = {}
        f = 0
        while len(pop.fronts[f]) != 0:
            for i in pop.fronts[f]:
                rank[i] = f
            f += 1
        for i, _ in enumerate(pop.crowding_distances):
            values.append({"index": i, "rank": rank.get(i, f), "distance": pop.crowding_distances[i]})
        return sorted(values, key=lambda k: (k["rank"], -k["distance"]))

    def fast_non_dominated_sort(self, pop: Population):
        self._calculate_dominants(pop)
        self._calculate_fronts(pop)

    def _evolve_to_offspring(self) -> Population:
        offspring = Population(self.core.population_size)
        offspring.population = []

        ordered = self._crowded_comparison_order(self.core.population_pt)
        fathers = [self.core.population_pt.population[i["index"]] for i in ordered]

        while len(offspring.population) < self.core.population_size:
            a = fathers[self.core.rnd_evol.randint(0, len(fathers) - 1)]
            b = fathers[self.core.rnd_evol.randint(0, len(fathers) - 1)]
            child = self.core.crossover(a, b)
            offspring.population.append(child)

        for i, child in enumerate(offspring.population):
            offspring.population[i] = self.core.mutate(child)

        self.core.calculate_population_fitness_objectives(offspring)
        return offspring

    def evolve(self):
        # Initialize if empty (THIS WILL NOW USE HEURISTIC INITIALIZATION)
        if not self.core.population_pt.population or len(self.core.population_pt.population[0]) == 0:
            self.core.generate_population(self.core.population_pt)

        self.fast_non_dominated_sort(self.core.population_pt)
        self._calculate_crowding(self.core.population_pt)

        total_generations = self.core.cfg.number_generations
        p_start = self.core.cfg.mutation_prob_start
        p_end = self.core.cfg.mutation_prob_end

        for gen in range(total_generations):
            # --- TÍCH HỢP ADAPTIVE MUTATION RATE ---
            # Tính toán xác suất mới dựa trên thế hệ hiện tại
            # Công thức giảm dần tuyến tính: High -> Low
            if total_generations > 1:
                new_prob = p_start - ((p_start - p_end) * (gen / (total_generations - 1)))
            else:
                new_prob = p_start
            
            # Cập nhật vào core để hàm mutate sử dụng
            self.core.current_mutation_prob = new_prob
            # ---------------------------------------

            pre = datetime.now()
            offspring = self._evolve_to_offspring()

            merged = self.core.population_pt.union(self.core.population_pt, offspring)
            self.fast_non_dominated_sort(merged)
            self._calculate_crowding(merged)
            ordered = self._crowded_comparison_order(merged)

            final = Population(self.core.population_size)
            final.population = []
            for i in range(self.core.population_size):
                final.population.append(merged.population[ordered[i]["index"]])
            self.core.calculate_population_fitness_objectives(final)

            self.core.population_pt = final
            self.fast_non_dominated_sort(self.core.population_pt)
            self._calculate_crowding(self.core.population_pt)

        return self.core.population_pt.pareto_export()