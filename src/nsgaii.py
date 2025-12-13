from datetime import datetime
from typing import List

from .population import Population
from .ga_core import GACore


class NSGAII:
    def __init__(self, core: GACore):
        self.core = core

    # NSGA-II helpers
    def _calculate_dominants(self, pop: Population):
        for i in range(len(pop.population)):
            pop.dominated_by[i] = set()
            pop.dominates_to[i] = set()
            pop.fronts[i] = set()

        def dominates(a, b):
            fa = pop.fitness[a]
            fb = pop.fitness[b]
            better_or_equal = (fa["latency"] <= fb["latency"]) and (fa["overload"] <= fb["overload"]) 
            strictly_better = (fa["latency"] < fb["latency"]) or (fa["overload"] < fb["overload"]) 
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

        for key in ("latency", "overload"):
            sorted_idx = sorted(front, key=lambda k: pop.fitness[k][key])
            pop.crowding_distances[sorted_idx[0]] = float("inf")
            pop.crowding_distances[sorted_idx[-1]] = float("inf")
            fmin = pop.fitness[sorted_idx[0]][key]
            fmax = pop.fitness[sorted_idx[-1]][key]
            denom = (fmax - fmin) if (fmax - fmin) != 0 else 1.0
            for k in range(1, len(sorted_idx) - 1):
                prev = pop.fitness[sorted_idx[k - 1]][key]
                nxt = pop.fitness[sorted_idx[k + 1]][key]
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

    # Evolution
    def _evolve_to_offspring(self) -> Population:
        offspring = Population(self.core.population_size)
        offspring.population = []

        ordered = self._crowded_comparison_order(self.core.population_pt)
        fathers = [self.core.population_pt.population[i["index"]] for i in ordered]

        # crossover until full
        while len(offspring.population) < self.core.population_size:
            a = fathers[self.core.rnd_evol.randint(0, len(fathers) - 1)]
            b = fathers[self.core.rnd_evol.randint(0, len(fathers) - 1)]
            child = self.core.crossover(a, b)
            offspring.population.append(child)

        # mutation
        for i, child in enumerate(offspring.population):
            offspring.population[i] = self.core.mutate(child)

        self.core.calculate_population_fitness_objectives(offspring)
        return offspring

    def evolve(self):
        # Initialize if empty
        if not self.core.population_pt.population or len(self.core.population_pt.population[0]) == 0:
            self.core.generate_population(self.core.population_pt)

        self.fast_non_dominated_sort(self.core.population_pt)
        self._calculate_crowding(self.core.population_pt)

        for _ in range(self.core.cfg.number_generations):
            pre = datetime.now()
            offspring = self._evolve_to_offspring()

            # Merge and select
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

        # Return Pareto front
        return self.core.population_pt.pareto_export()
