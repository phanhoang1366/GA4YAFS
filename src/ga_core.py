import random
import math
from typing import Dict, Any, List

import numpy as np

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
        
        # Objective weights (equal by default; can be tuned)
        self.latency_weight = 1.0 / 3.0
        self.spread_weight = 1.0 / 3.0
        self.resource_weight = 1.0 / 3.0

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
        """
        Apply mutation with multiple operators for diverse exploration.
        Operators:
        1. randomAssignment: randomly reassign services to nodes
        2. serviceShuffle: swap placements of two random services
        3. neighborSwap: move a service to a neighbor node in the topology
        """
        mutation_type = self.rnd_evol.choice(['random', 'shuffle', 'neighbor'])
        newc = chrom[:]
        
        if mutation_type == 'random':
            # Random reassignment: reassign some services randomly
            for i in range(len(newc)):
                if self.rnd_evol.random() < self.cfg.mutation_probability:
                    newc[i] = self.rnd_evol.randint(0, len(self.system.fog_nodes) - 1)
        
        elif mutation_type == 'shuffle':
            # Service shuffle: swap placements of two random services
            if len(newc) > 1:
                for _ in range(max(1, int(self.cfg.mutation_probability * len(newc)))):
                    i = self.rnd_evol.randint(0, len(newc) - 1)
                    j = self.rnd_evol.randint(0, len(newc) - 1)
                    newc[i], newc[j] = newc[j], newc[i]
        
        elif mutation_type == 'neighbor':
            # Neighbor swap: move services to topologically nearby nodes
            if self.system.G is None:
                # Fallback to random mutation if topology graph not available
                for i in range(len(newc)):
                    if self.rnd_evol.random() < self.cfg.mutation_probability:
                        newc[i] = self.rnd_evol.randint(0, len(self.system.fog_nodes) - 1)
            else:
                for i in range(len(newc)):
                    if self.rnd_evol.random() < self.cfg.mutation_probability:
                        current_node = newc[i]
                        # Get neighbors in topology
                        neighbors = list(self.system.G.neighbors(self.system.fog_nodes[current_node]))
                        if neighbors:
                            # Convert neighbor node ID to fog_node index
                            neighbor_indices = [self.system.fog_nodes.index(n) for n in neighbors if n in self.system.fog_nodes]
                            if neighbor_indices:
                                newc[i] = self.rnd_evol.choice(neighbor_indices)
                            else:
                                # Fallback: random reassignment if no valid neighbor index
                                newc[i] = self.rnd_evol.randint(0, len(self.system.fog_nodes) - 1)
                        else:
                            # Fallback: random reassignment if node has no neighbors
                            newc[i] = self.rnd_evol.randint(0, len(self.system.fog_nodes) - 1)
        
        return newc

    def crossover(self, a: List[int], b: List[int]) -> List[int]:
        """
                Multi-point crossover closer to original GAcore:
                - If service grouping available, perform group-aware crossover:
                    choose a single cut per app-group (like original MIO2 per row) and
                    alternate parent segments per group.
                - Otherwise, choose multiple cut points across the whole chromosome
                    and alternate segments from parent A and B.
        """
        la = len(a)
        lb = len(b)
        if la == 0 and lb == 0:
            return []
        if la != lb:
            # Fallback: copy the shorter prefix from A then fill with B
            m = min(la, lb)
            child = (a[:m] if self.rnd_evol.random() < 0.5 else b[:m])
            base = a if len(a) >= len(b) else b
            child += base[m:]
            return child
        n = la
        if n <= 2:
            # Small chromosomes: uniform per-gene selection
            return [a[i] if self.rnd_evol.random() < 0.5 else b[i] for i in range(n)]

        # If we have service_modules, build contiguous app groups
        groups = []
        try:
            mods = getattr(self.system, 'service_modules', [])
            if mods and len(mods) == n:
                current_app = mods[0]['app_id']
                start_idx = 0
                for i in range(1, n):
                    if mods[i]['app_id'] != current_app:
                        groups.append((start_idx, i))  # [start, end) slice
                        start_idx = i
                        current_app = mods[i]['app_id']
                groups.append((start_idx, n))
        except Exception:
            groups = []

        child: List[int] = []
        if groups and len(groups) > 0:
            # Group-aware crossover: choose one cut inside each group and alternate parents per group
            take_from_a = self.rnd_evol.random() < 0.5
            for (g_start, g_end) in groups:
                g_len = g_end - g_start
                if g_len <= 1:
                    # Single gene group: pick per-gene
                    gene = a[g_start] if take_from_a else b[g_start]
                    child.append(gene)
                else:
                    # Choose a single cut point inside the group (like MIO2)
                    cut = self.rnd_evol.randint(g_start + 1, g_end - 1)
                    seg1 = (a[g_start:cut] if take_from_a else b[g_start:cut])
                    seg2 = (b[cut:g_end] if take_from_a else a[cut:g_end])
                    child.extend(seg1)
                    child.extend(seg2)
                # Alternate parent choice for next group
                take_from_a = not take_from_a
            return child
        else:
            # Fallback: whole-chromosome multi-cut alternating segments
            max_cuts = 5 if n >= 10 else (3 if n >= 5 else 2)
            num_cuts = self.rnd_evol.randint(2, max_cuts)
            cut_indices = sorted(set(self.rnd_evol.sample(range(1, n), k=num_cuts)))
            boundaries = [0] + cut_indices + [n]

            take_from_a = self.rnd_evol.random() < 0.5
            for i in range(len(boundaries) - 1):
                start = boundaries[i]
                end = boundaries[i + 1]
                segment = (a[start:end] if take_from_a else b[start:end])
                child.extend(segment)
                take_from_a = not take_from_a
            return child

    def calculate_population_fitness_objectives(self, pop: Population):
        """
        Three objectives (all to minimize via NSGA-II):
        1) Latency: network delay between service placements
        2) Spread: coefficient of variation (lower = more balanced distribution)
        3) Resource Underutilization: 1 - utilization (lower = higher utilization)
        """
        for idx, chrom in enumerate(pop.population):
            latency = self._calculate_latency(chrom)
            spread = self._calculate_spread(chrom)
            utilization = self._calculate_resource_utilization(chrom)
            underutilization = 1.0 - utilization  # Minimize underutilization = maximize utilization
            
            pop.fitness[idx] = {
                "latency": latency,
                "spread": spread,
                "underutilization": underutilization,
                "index": idx
            }
    
    def _calculate_latency(self, chrom: List[int]) -> float:
        """
        Objective 1: Minimize network latency based on service dependency DAG.
        For each service, sum distances to its dependent services.
        """
        if len(chrom) == 0 or len(self.system.service_matrix) == 0:
            return 0.0
        
        total_latency = 0.0
        num_dependencies = 0
        
        # For each service, check its dependencies in the service matrix
        for src_service_idx in range(len(chrom)):
            src_node_idx = chrom[src_service_idx]
            for dst_service_idx in range(len(chrom)):
                # If service src depends on service dst
                if self.system.service_matrix[src_service_idx][dst_service_idx] == 1:
                    dst_node_idx = chrom[dst_service_idx]
                    total_latency += self.system.dev_distance_matrix[src_node_idx][dst_node_idx]
                    num_dependencies += 1
        
        # Normalize by average path length
        if num_dependencies > 0:
            normalized_latency = total_latency / (num_dependencies * self.system.average_path_length)
        else:
            normalized_latency = 0.0
        
        return normalized_latency
    
    def _calculate_spread(self, chrom: List[int]) -> float:
        """
        Objective 2: Maximize even distribution of services across fog devices.
        Measures balance using coefficient of variation of placement counts and
        average distance variance between placed services (approximation of spread).
        Lower spread = more balanced and geographically distributed placement.
        """
        if len(chrom) == 0:
            return 0.0
        
        # Metric 1: Coefficient of variation of placement counts per fog node
        placement_count = [0 for _ in self.system.fog_nodes]
        for fog_idx in chrom:
            placement_count[fog_idx] += 1
        
        non_zero_counts = [c for c in placement_count if c > 0]
        if len(non_zero_counts) == 0:
            return 1.0  # All services unplaced (worst case)
        
        mean_count = np.mean(non_zero_counts)
        std_count = np.std(non_zero_counts)
        count_variance = (std_count / mean_count) if mean_count > 0 else 0.0
        
        # Metric 2: Average pairwise distance between all placed services
        # (approximates geographic spread; higher distance = more spread out = better)
        distance_variance = 0.0
        placed_nodes = [chrom[s_idx] for s_idx in range(len(chrom))]
        if len(placed_nodes) > 1:
            pairwise_distances = []
            for i in range(len(placed_nodes)):
                for j in range(i + 1, len(placed_nodes)):
                    dist = self.system.dev_distance_matrix[placed_nodes[i]][placed_nodes[j]]
                    pairwise_distances.append(dist)
            
            if pairwise_distances:
                mean_dist = np.mean(pairwise_distances)
                # Find max distance in the distance matrix
                max_possible_dist = 0.0
                for row in self.system.dev_distance_matrix:
                    for val in row:
                        if val > max_possible_dist:
                            max_possible_dist = val
                
                # Normalize to [0, 1] where higher = more spread out
                if max_possible_dist > 0:
                    normalized_mean_dist = mean_dist / max_possible_dist
                    distance_variance = 1.0 - normalized_mean_dist  # Invert: lower metric = better
                else:
                    distance_variance = 0.0
        
        # Combine metrics: balance (count variance) + compactness (distance variance)
        # Both components favor even distribution: low count variance + high distance variance
        spread_score = 0.6 * count_variance + 0.4 * distance_variance
        return float(spread_score)
    
    def _calculate_resource_utilization(self, chrom: List[int]) -> float:
        """
        Objective 3: Maximize fog resource utilization (minimize unused resources).
        Returns: fraction of fog resources actually used (0-1, higher is better).
        """
        used = [0 for _ in self.system.fog_nodes]
        for s_idx, fog_idx in enumerate(chrom):
            used[fog_idx] += self.system.service_resources[s_idx]
        
        total_used = sum(used)
        total_available = sum(self.system.fog_resources)
        
        if total_available == 0:
            return 0.0
        
        utilization = total_used / total_available
        return utilization  # Higher is better

    def is_feasible(self, chrom: List[int]) -> bool:
        """
        Constraint validation: Check if placement respects fog node resource limits.
        Returns True if placement is feasible (all nodes have sufficient resources).
        """
        if len(chrom) != len(self.system.service_resources):
            return False
        
        used = [0 for _ in self.system.fog_nodes]
        for s_idx, fog_idx in enumerate(chrom):
            if fog_idx < 0 or fog_idx >= len(self.system.fog_nodes):
                return False
            used[fog_idx] += self.system.service_resources[s_idx]
        
        # Check each node's resource constraint
        for node_idx, resources_used in enumerate(used):
            if resources_used > self.system.fog_resources[node_idx]:
                return False
        
        return True

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
