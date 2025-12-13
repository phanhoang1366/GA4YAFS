import random
from typing import Dict, Any, List

from .population import Population
from .system_model import SystemModel
from .config import Config


class GACore:
    def __init__(self, system: SystemModel, cfg: Config):
        self.system = system
        self.cfg = cfg
        self.rnd_pop = random.Random(cfg.population_seed)
        self.rnd_evol = random.Random(cfg.evolution_seed)
        self.population_size = cfg.population_size

        self.population_pt = Population(self.population_size)

    # Chromosome representation: list of length service_number
    # Each position maps service(module) i -> chosen fog node index
    def generate_population(self, pop: Population):
        pop.population = []
        for _ in range(self.population_size):
            chrom = [self.rnd_pop.randint(0, len(self.system.fog_nodes) - 1)
                     for _ in range(self.system.service_number)]
            pop.population.append(chrom)
        # initialize metadata
        pop.fitness = [{} for _ in range(len(pop.population))]
        pop.dominates_to = [set() for _ in range(len(pop.population))]
        pop.dominated_by = [set() for _ in range(len(pop.population))]
        pop.fronts = [set() for _ in range(len(pop.population))]
        pop.crowding_distances = [0.0 for _ in range(len(pop.population))]
        self.calculate_population_fitness_objectives(pop)

    def mutate(self, chrom: List[int]) -> List[int]:
        newc = chrom[:]
        for i in range(len(newc)):
            if self.rnd_evol.random() < self.cfg.mutation_probability:
                newc[i] = self.rnd_evol.randint(0, len(self.system.fog_nodes) - 1)
        return newc

    def crossover(self, a: List[int], b: List[int]) -> List[int]:
        if len(a) <= 1 or len(b) <= 1:
            return a[:] if self.rnd_evol.random() < 0.5 else b[:]
        cut = self.rnd_evol.randint(1, len(a) - 1)
        return a[:cut] + b[cut:]

    def calculate_population_fitness_objectives(self, pop: Population):
        # Two objectives (to minimize):
        # 1) total latency: sum of shortest-path latencies from service placements pairwise
        # 2) resource overload: sum of overload across fog nodes (assignments exceeding capacity)
        for idx, chrom in enumerate(pop.population):
            # Objective 1: latency among services (approximate: sum distances to first service)
            if len(chrom) > 0:
                first = chrom[0]
                lat_sum = 0.0
                for s in chrom[1:]:
                    lat_sum += self.system.dev_distance_matrix[first][s]
            else:
                lat_sum = 0.0

            # Objective 2: overload
            used = [0 for _ in self.system.fog_nodes]
            for s_idx, fog_idx in enumerate(chrom):
                used[fog_idx] += self.system.service_resources[s_idx]
            overload = 0
            for i, u in enumerate(used):
                capacity = self.system.fog_resources[i]
                if u > capacity:
                    overload += (u - capacity)

            pop.fitness[idx] = {"latency": lat_sum, "overload": float(overload), "index": idx}

    def chromosome_to_placement_json(self, chrom: List[int]) -> Dict[str, Any]:
        # Produce allocDefinition-like entries mapping actual modules to fog nodes
        # Compatible with JSONPlacement: list of dicts with module_name, app, id_resource
        placement = []
        for s_idx, fog_idx in enumerate(chrom):
            node_id = self.system.fog_nodes[fog_idx]
            mod = self.system.service_modules[s_idx]
            placement.append({
                "module_name": mod["module_name"],
                "app": mod["app_id"],
                "id_resource": node_id
            })
        return {"initialAllocation": placement}
