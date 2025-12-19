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

    # Chromosome representation: 2D array with 2 rows
    # Row 0: Primary node placement for each service (fog node index)
    # Row 1: Replication factor for each service (1 = no replicas, 2 = 1 additional, etc.)
    def generate_population(self, pop: Population):
        pop.population = []
        for _ in range(self.population_size):
            placement_row = [self.rnd_pop.randint(0, len(self.system.fog_nodes) - 1)
                           for _ in range(self.system.service_number)]
            replica_row = [self.rnd_pop.randint(1, self.cfg.max_replicas)
                          for _ in range(self.system.service_number)]
            chrom = [placement_row, replica_row]
            pop.population.append(chrom)
        # initialize metadata
        pop.fitness = [{} for _ in range(len(pop.population))]
        pop.dominates_to = [set() for _ in range(len(pop.population))]
        pop.dominated_by = [set() for _ in range(len(pop.population))]
        pop.fronts = [set() for _ in range(len(pop.population))]
        pop.crowding_distances = [0.0 for _ in range(len(pop.population))]
        self.calculate_population_fitness_objectives(pop)

    def mutate(self, chrom: List[List[int]]) -> List[List[int]]:
        """
        Apply mutation to 2-row chromosome.
        Row 0 (placement): random/shuffle/neighbor mutations
        Row 1 (replicas): adjust replica counts
        """
        mutation_type = self.rnd_evol.choice(['random', 'shuffle', 'neighbor', 'replica_adjust'])
        newc = [chrom[0][:], chrom[1][:]]
        
        if mutation_type == 'random':
            # Random reassignment: reassign some services randomly
            for i in range(len(newc[0])):
                if self.rnd_evol.random() < self.cfg.mutation_probability:
                    newc[0][i] = self.rnd_evol.randint(0, len(self.system.fog_nodes) - 1)
        
        elif mutation_type == 'shuffle':
            # Service shuffle: swap placements and replicas of two random services
            if len(newc[0]) > 1:
                for _ in range(max(1, int(self.cfg.mutation_probability * len(newc[0])))):
                    i = self.rnd_evol.randint(0, len(newc[0]) - 1)
                    j = self.rnd_evol.randint(0, len(newc[0]) - 1)
                    newc[0][i], newc[0][j] = newc[0][j], newc[0][i]
                    newc[1][i], newc[1][j] = newc[1][j], newc[1][i]
        
        elif mutation_type == 'neighbor':
            # Neighbor swap: move services to topologically nearby nodes
            if self.system.G is None:
                for i in range(len(newc[0])):
                    if self.rnd_evol.random() < self.cfg.mutation_probability:
                        newc[0][i] = self.rnd_evol.randint(0, len(self.system.fog_nodes) - 1)
            else:
                for i in range(len(newc[0])):
                    if self.rnd_evol.random() < self.cfg.mutation_probability:
                        current_node = newc[0][i]
                        neighbors = list(self.system.G.neighbors(self.system.fog_nodes[current_node]))
                        if neighbors:
                            neighbor_indices = [self.system.fog_nodes.index(n) for n in neighbors if n in self.system.fog_nodes]
                            if neighbor_indices:
                                newc[0][i] = self.rnd_evol.choice(neighbor_indices)
                            else:
                                newc[0][i] = self.rnd_evol.randint(0, len(self.system.fog_nodes) - 1)
                        else:
                            newc[0][i] = self.rnd_evol.randint(0, len(self.system.fog_nodes) - 1)
        
        elif mutation_type == 'replica_adjust':
            # Adjust replica counts: increment or decrement
            for i in range(len(newc[1])):
                if self.rnd_evol.random() < self.cfg.mutation_probability:
                    delta = self.rnd_evol.choice([-1, 0, 1])
                    newc[1][i] = max(1, min(self.cfg.max_replicas, newc[1][i] + delta))
        
        return newc

    def crossover(self, a: List[List[int]], b: List[List[int]]) -> List[List[int]]:
        """
        Synchronized crossover for 2-row chromosome:
        - Same cut point for both placement and replica rows
        - Maintains correlation between placement and replica count
        """
        # Extract rows
        a_placement, a_replicas = a[0], a[1]
        b_placement, b_replicas = b[0], b[1]
        
        la = len(a_placement)
        lb = len(b_placement)
        
        if la == 0 and lb == 0:
            return [[], []]
        if la != lb:
            # Fallback: copy from parent with more services
            if la >= lb:
                return [a_placement[:], a_replicas[:]]
            else:
                return [b_placement[:], b_replicas[:]]
        
        n = la
        if n <= 2:
            # Small chromosomes: uniform per-gene selection
            child_placement = [a_placement[i] if self.rnd_evol.random() < 0.5 else b_placement[i] for i in range(n)]
            child_replicas = [a_replicas[i] if self.rnd_evol.random() < 0.5 else b_replicas[i] for i in range(n)]
            return [child_placement, child_replicas]

        # Single-point synchronized crossover
        cut = self.rnd_evol.randint(1, n - 1)
        child_placement = a_placement[:cut] + b_placement[cut:]
        child_replicas = a_replicas[:cut] + b_replicas[cut:]
        
        return [child_placement, child_replicas]

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
    
    def _calculate_latency(self, chrom: List[List[int]]) -> float:
        """
        Objective 1: Minimize network latency based on service dependency DAG.
        For 2-row chromosome, use primary placement (row 0) for distance calculations.
        """
        placement_row = chrom[0]
        if len(placement_row) == 0 or len(self.system.service_matrix) == 0:
            return 0.0
        
        total_latency = 0.0
        num_dependencies = 0
        
        # For each service, check its dependencies in the service matrix
        for src_service_idx in range(len(placement_row)):
            src_node_idx = placement_row[src_service_idx]
            for dst_service_idx in range(len(placement_row)):
                # If service src depends on service dst
                if self.system.service_matrix[src_service_idx][dst_service_idx] == 1:
                    dst_node_idx = placement_row[dst_service_idx]
                    total_latency += self.system.dev_distance_matrix[src_node_idx][dst_node_idx]
                    num_dependencies += 1
        
        # Normalize by average path length
        if num_dependencies > 0:
            normalized_latency = total_latency / (num_dependencies * self.system.average_path_length)
        else:
            normalized_latency = 0.0
        
        return normalized_latency
    
    def _calculate_spread(self, chrom: List[List[int]]) -> float:
        """
        Objective 2: Maximize even distribution of services across fog devices.
        For 2-row chromosome, considers both placement and replica counts.
        Lower spread = more balanced and geographically distributed placement.
        """
        placement_row = chrom[0]
        replica_row = chrom[1]
        
        if len(placement_row) == 0:
            return 0.0
        
        # Metric 1: Coefficient of variation of placement counts per fog node
        # Account for replicas: each service contributes replica_count instances
        placement_count = [0 for _ in self.system.fog_nodes]
        for s_idx, fog_idx in enumerate(placement_row):
            placement_count[fog_idx] += replica_row[s_idx]
        
        non_zero_counts = [c for c in placement_count if c > 0]
        if len(non_zero_counts) == 0:
            return 1.0  # All services unplaced (worst case)
        
        mean_count = np.mean(non_zero_counts)
        std_count = np.std(non_zero_counts)
        count_variance = (std_count / mean_count) if mean_count > 0 else 0.0
        
        # Metric 2: Average pairwise distance between all placed services
        # (approximates geographic spread; higher distance = more spread out = better)
        distance_variance = 0.0
        placed_nodes = [placement_row[s_idx] for s_idx in range(len(placement_row))]
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
    
    def _calculate_resource_utilization(self, chrom: List[List[int]]) -> float:
        """
        Objective 3: Maximize fog resource utilization (minimize unused resources).
        For 2-row chromosome, accounts for replica counts.
        Returns: fraction of fog resources actually used (0-1, higher is better).
        """
        placement_row = chrom[0]
        replica_row = chrom[1]
        
        # Account for primary placement only (replicas placed via strategy, not counted in fitness)
        # This simplification avoids double-counting and complex replica placement in fitness
        used = [0 for _ in self.system.fog_nodes]
        for s_idx, fog_idx in enumerate(placement_row):
            # Only count primary placement for utilization metric
            used[fog_idx] += self.system.service_resources[s_idx]
        
        total_used = sum(used)
        total_available = sum(self.system.fog_resources)
        
        if total_available == 0:
            return 0.0
        
        utilization = total_used / total_available
        return utilization  # Higher is better

    def is_feasible(self, chrom: List[List[int]]) -> bool:
        """
        Constraint validation: Check if placement respects fog node resource limits.
        For 2-row chromosome, considers primary placement only (replicas validated at export).
        Returns True if placement is feasible (all nodes have sufficient resources).
        """
        placement_row = chrom[0]
        replica_row = chrom[1]
        
        if len(placement_row) != len(self.system.service_resources):
            return False
        if len(replica_row) != len(self.system.service_resources):
            return False
        
        # Check primary placements only for feasibility (conservative check)
        used = [0 for _ in self.system.fog_nodes]
        for s_idx, fog_idx in enumerate(placement_row):
            if fog_idx < 0 or fog_idx >= len(self.system.fog_nodes):
                return False
            # Count primary + replicas for pessimistic feasibility
            # Assume all replicas might end up on same node (worst case)
            used[fog_idx] += self.system.service_resources[s_idx] * replica_row[s_idx]
        
        # Check each node's resource constraint
        for node_idx, resources_used in enumerate(used):
            if resources_used > self.system.fog_resources[node_idx]:
                return False
        
        return True

    def chromosome_to_placement_json(self, chrom: List[List[int]]) -> Dict[str, Any]:
        """
        Export 2-row chromosome to YAFS placement JSON.
        Primary placements from row 0, replicas placed using nearest-neighbor strategy.
        """
        placement_row = chrom[0]
        replica_row = chrom[1]
        placement = []
        
        for s_idx in range(len(placement_row)):
            primary_fog_idx = placement_row[s_idx]
            num_replicas = replica_row[s_idx]
            mod = self.system.service_modules[s_idx]
            
            # Place primary instance
            placement.append({
                "module_name": mod["module_name"],
                "app": mod["app_id"],
                "id_resource": self.system.fog_nodes[primary_fog_idx]
            })
            
            # Place additional replicas using nearest-neighbor strategy
            if num_replicas > 1:
                replica_nodes = self._get_replica_nodes(primary_fog_idx, num_replicas - 1, s_idx)
                for rep_fog_idx in replica_nodes:
                    placement.append({
                        "module_name": mod["module_name"],
                        "app": mod["app_id"],
                        "id_resource": self.system.fog_nodes[rep_fog_idx]
                    })
        
        return {"initialAllocation": placement}
    
    def _get_replica_nodes(self, primary_node_idx: int, num_replicas: int, service_idx: int) -> List[int]:
        """
        Select nodes for replicas using nearest-neighbor strategy.
        Prioritizes topologically close nodes with available resources.
        """
        if num_replicas <= 0:
            return []
        
        replica_nodes = []
        candidates = list(range(len(self.system.fog_nodes)))
        candidates.remove(primary_node_idx)  # Exclude primary node
        
        # Sort candidates by distance from primary node
        if self.system.G is not None:
            # Use topology distance if available
            try:
                import networkx as nx
                primary_id = self.system.fog_nodes[primary_node_idx]
                distances = []
                for cand_idx in candidates:
                    cand_id = self.system.fog_nodes[cand_idx]
                    try:
                        dist = nx.shortest_path_length(self.system.G, primary_id, cand_id)
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        dist = float('inf')
                    distances.append((dist, cand_idx))
                distances.sort()
                candidates = [idx for _, idx in distances]
            except:
                pass  # Fallback to unsorted candidates
        
        # Select nearest neighbors with resource checks
        service_resource = self.system.service_resources[service_idx]
        for cand_idx in candidates:
            if len(replica_nodes) >= num_replicas:
                break
            # Simple capacity check (optimistic: assume node has space)
            if self.system.fog_resources[cand_idx] >= service_resource:
                replica_nodes.append(cand_idx)
        
        # If not enough valid candidates, fill with remaining nodes (best-effort)
        while len(replica_nodes) < num_replicas and len(candidates) > len(replica_nodes):
            for cand_idx in candidates:
                if cand_idx not in replica_nodes:
                    replica_nodes.append(cand_idx)
                    if len(replica_nodes) >= num_replicas:
                        break
        
        return replica_nodes[:num_replicas]
