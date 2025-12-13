import copy


class Population:
    def __init__(self, size):
        self.population = [list()] * size
        self.fitness = [{}] * size
        self.fitness_normalized = [{}] * size
        self.dominates_to = [set()] * size
        self.dominated_by = [set()] * size
        self.fronts = [set()] * size
        self.crowding_distances = [float(0)] * size

    def union(self, a, b):
        r = Population(1)
        r.population = copy.deepcopy(a.population) + copy.deepcopy(b.population)
        r.fitness = copy.deepcopy(a.fitness) + copy.deepcopy(b.fitness)
        r.fitness_normalized = copy.deepcopy(a.fitness_normalized) + copy.deepcopy(b.fitness_normalized)
        for i, _ in enumerate(r.fitness):
            r.fitness[i]["index"] = i
        r.dominates_to = copy.deepcopy(a.dominates_to) + copy.deepcopy(b.dominates_to)
        r.dominated_by = copy.deepcopy(a.dominated_by) + copy.deepcopy(b.dominated_by)
        r.fronts = copy.deepcopy(a.fronts) + copy.deepcopy(b.fronts)
        r.crowding_distances = copy.deepcopy(a.crowding_distances) + copy.deepcopy(b.crowding_distances)
        return r

    def pareto_export(self):
        pareto = Population(len(self.population))
        for i in self.fronts[0]:
            pareto.population[i] = self.population[i]
            pareto.fitness[i] = self.fitness[i]
        return pareto

    def replace_solution(self, new_pop, new_id, old_pop, old_id):
        old_pop.population[old_id] = new_pop.population[new_id]
        old_pop.fitness[old_id] = new_pop.fitness[new_id]
        old_pop.fitness_normalized[old_id] = new_pop.fitness_normalized[new_id]
        old_pop.dominates_to[old_id] = new_pop.dominates_to[new_id]
        old_pop.dominated_by[old_id] = new_pop.dominated_by[new_id]
        old_pop.fronts[old_id] = new_pop.fronts[new_id]
        old_pop.crowding_distances[old_id] = new_pop.crowding_distances[new_id]

    def remove(self, idx):
        del self.population[idx]
        del self.fitness[idx]
        del self.fitness_normalized[idx]
        del self.dominates_to[idx]
        del self.dominated_by[idx]
        del self.fronts[idx]
        del self.crowding_distances[idx]

    def append_and_increase(self, solution):
        self.population.append(solution)
        self.fitness = self.fitness + [{}]
        self.fitness_normalized = self.fitness_normalized + [{}]
        self.dominates_to = self.dominates_to + [set()]
        self.dominated_by = self.dominated_by + [set()]
        self.fronts = self.fronts + [set()]
        self.crowding_distances = self.crowding_distances + [float(0)]
