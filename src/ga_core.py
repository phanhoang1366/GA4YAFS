import random
import math
import hashlib
from typing import Dict, Any, List, Optional

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
        # Auto-adjust flag: when True, scale spread magnitudes to be comparable
        # with latency based on population observations before computing a
        # weighted `total` score. Does not change individual objectives used
        # by NSGA-II, but provides a `total` for convenience/legacy code.
        self.auto_adjust_weights = getattr(cfg, "auto_adjust_weights", False)

    def _chrom_seed(self, placement_row: List[int], replica_row: List[int], extra: str = "") -> int:
        """Deterministic seed for a chromosome + optional extra string."""
        s1 = ",".join(str(x) for x in placement_row)
        s2 = ",".join(str(x) for x in replica_row)
        blob = (s1 + "|" + s2 + "|" + extra).encode("utf-8")
        h = hashlib.sha256(blob).digest()
        # use first 8 bytes to form an int seed
        return int.from_bytes(h[:8], "big")

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
        latencies = []
        spreads = []
        unders = []

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

            latencies.append(latency)
            spreads.append(spread)
            unders.append(underutilization)

        # Compute automatic scaling factor for spread -> latency magnitude
        beta = 1.0
        if self.auto_adjust_weights and len(spreads) > 0:
            spread_max = max(spreads)
            latency_min = min(latencies) if len(latencies) > 0 else 0.0
            if spread_max > 0:
                beta = (latency_min / spread_max) if latency_min > 0 else 1.0
            else:
                beta = 1.0

        # Attach a convenience `total` score and scaled components so legacy
        # code or callers that expect a single scalar can use it. `total`
        # respects the configured objective weights but uses an automatically
        # scaled spread when `auto_adjust_weights` is enabled.
        for idx in range(len(pop.population)):
            f = pop.fitness[idx]
            norm_spread = f["spread"] * beta
            total = (
                self.latency_weight * f["latency"]
                + self.spread_weight * norm_spread
                + self.resource_weight * f["underutilization"]
            )
            f["total"] = total
            f["wlat"] = f["latency"]
            f["wsp"] = norm_spread
            f["wres"] = f["underutilization"]
            pop.fitness[idx] = f
            
    def _calculate_latency(self, chrom: List[List[int]]) -> float:
        """
        Objective 1: Minimize network latency based on service dependency DAG.
        Accounts for replicas: for each service interaction, finds the minimum
        distance to any replica of the dependency (mimics original GA4YAFS).
        """
        placement_row = chrom[0]
        replica_row = chrom[1]
        
        if len(placement_row) == 0 or len(self.system.service_matrix) == 0:
            return 0.0
        
        # Get replica placements for all services (cached for efficiency)
        service_replica_nodes = {}
        for s_idx in range(len(placement_row)):
            primary_node = placement_row[s_idx]
            num_replicas = replica_row[s_idx]
            
            replica_nodes = [primary_node]
            if num_replicas > 1:
                seed = self._chrom_seed(placement_row, replica_row, extra=f"srv{s_idx}")
                srv_rng = random.Random(seed)
                additional_replicas = self._get_replica_nodes(primary_node, num_replicas - 1, s_idx, rng=srv_rng)
                replica_nodes.extend(additional_replicas)
            
            service_replica_nodes[s_idx] = replica_nodes
        
        total_distance = 0.0
        num_interactions = 0
        
        # For each service that has dependencies
        for src_service in range(len(placement_row)):
            consumed_services = []
            for dst_service in range(len(self.system.service_matrix[src_service])):
                if self.system.service_matrix[src_service][dst_service] == 1:
                    consumed_services.append(dst_service)
            
            if len(consumed_services) > 0:
                src_nodes = service_replica_nodes[src_service]
                
                # For each consumed service, find minimum distance from any src replica to any dst replica
                for dst_service in consumed_services:
                    dst_nodes = service_replica_nodes[dst_service]
                    
                    # For each instance of the source service, find nearest instance of dependency
                    for src_node in src_nodes:
                        min_dist = float('inf')
                        for dst_node in dst_nodes:
                            dist = self.system.dev_distance_matrix[src_node][dst_node]
                            min_dist = min(min_dist, dist)
                        total_distance += min_dist
                        num_interactions += 1
        
        # Calculate average distance per interaction
        if num_interactions > 0:
            avg_distance = total_distance / num_interactions
            # Normalize by average path length
            normalized_latency = avg_distance / self.system.average_path_length
        else:
            normalized_latency = 0.0
        
        return normalized_latency
    
    def _calculate_spread(self, chrom: List[List[int]]) -> float:
        """
        Objective 2: Spread metric with two modes:
        - If replicas are disabled (`max_replicas == 1`): use global balance
          + compactness across all services (counts + placement distances).
        - Otherwise: per-service replica CV (original GA4YAFS semantics).
        Lower is better in both cases.
        """
        placement_row = chrom[0]
        replica_row = chrom[1]
        
        if len(placement_row) == 0:
            return 0.0

        # Mode 1: replicas disabled → global spread
        if self.cfg.max_replicas == 1:
            # Coefficient of variation of service counts per fog node
            placement_count = [0 for _ in self.system.fog_nodes]
            for fog_idx in placement_row:
                placement_count[fog_idx] += 1

            non_zero_counts = [c for c in placement_count if c > 0]
            if len(non_zero_counts) == 0:
                return 1.0

            mean_count = np.mean(non_zero_counts)
            std_count = np.std(non_zero_counts)
            count_variance = (std_count / mean_count) if mean_count > 0 else 0.0

            # Compactness: average pairwise distance between placed services
            compactness_score = 0.0
            placed_nodes = placement_row[:]  # one entry per service
            if len(placed_nodes) > 1:
                pairwise_distances = []
                for i in range(len(placed_nodes)):
                    for j in range(i + 1, len(placed_nodes)):
                        dist = self.system.dev_distance_matrix[placed_nodes[i]][placed_nodes[j]]
                        pairwise_distances.append(dist)

                if pairwise_distances:
                    mean_dist = np.mean(pairwise_distances)
                    max_possible_dist = 0.0
                    for row in self.system.dev_distance_matrix:
                        for val in row:
                            if val > max_possible_dist:
                                max_possible_dist = val

                    if max_possible_dist > 0:
                        normalized_mean_dist = mean_dist / max_possible_dist
                        compactness_score = 1.0 - normalized_mean_dist  # more spread → lower compactness
                    else:
                        compactness_score = 0.0

            # Balance both components (lower is better)
            return float(0.5 * count_variance + 0.5 * compactness_score)

        # Mode 2: replicas enabled → per-service replica CV
        total_spread = 0.0

        for s_idx in range(len(placement_row)):
            primary_node = placement_row[s_idx]
            num_replicas = replica_row[s_idx]

            replica_nodes = [primary_node]
            if num_replicas > 1:
                seed = self._chrom_seed(placement_row, replica_row, extra=f"srv{s_idx}")
                srv_rng = random.Random(seed)
                additional_replicas = self._get_replica_nodes(primary_node, num_replicas - 1, s_idx, rng=srv_rng)
                replica_nodes.extend(additional_replicas)

            if len(replica_nodes) > 1:
                pairwise_distances = []
                for i in range(len(replica_nodes)):
                    for j in range(i + 1, len(replica_nodes)):
                        dist = self.system.dev_distance_matrix[replica_nodes[i]][replica_nodes[j]]
                        pairwise_distances.append(dist)

                if pairwise_distances and np.mean(pairwise_distances) > 0:
                    mean_dist = np.mean(pairwise_distances)
                    std_dist = np.std(pairwise_distances)
                    service_spread = std_dist / mean_dist
                else:
                    service_spread = 0.0
            else:
                service_spread = 1.0

            total_spread += service_spread

        return float(total_spread / len(placement_row))
    
    def _calculate_resource_utilization(self, chrom: List[List[int]]) -> float:
        """
        Objective 3: Maximize fog resource utilization (minimize unused resources).
        For 2-row chromosome, accounts for replica counts.
        Returns: fraction of fog resources actually used (0-1, higher is better).
        """
        placement_row = chrom[0]
        replica_row = chrom[1]

        # Conservative but placement-aware accounting:
        # count primary instance on its node, and allocate each additional replica
        # to the nodes selected by `_get_replica_nodes()` so utilization depends
        # on how replicas are distributed.
        used = [0 for _ in self.system.fog_nodes]
        # Cache replica selections per service to avoid repeated path computations
        replica_selection_cache = {}
        # Deterministic per-chromosome RNG to make random replica placement repeatable
        seed = self._chrom_seed(placement_row, replica_row)
        chrom_rng = random.Random(seed)
        for s_idx, primary_idx in enumerate(placement_row):
            res = self.system.service_resources[s_idx]
            # primary
            if 0 <= primary_idx < len(used):
                used[primary_idx] += res
            # replicas
            n_reps = max(0, replica_row[s_idx] - 1)
            if n_reps > 0:
                if s_idx in replica_selection_cache:
                    rep_nodes = replica_selection_cache[s_idx]
                else:
                    rep_nodes = self._get_replica_nodes(primary_idx, n_reps, s_idx, rng=chrom_rng)
                    replica_selection_cache[s_idx] = rep_nodes
                for rn in rep_nodes:
                    if 0 <= rn < len(used):
                        used[rn] += res
        
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
        
        # Use deterministic RNG per-chromosome so exported replica placement is stable
        seed = self._chrom_seed(placement_row, replica_row)
        chrom_rng = random.Random(seed)

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
                replica_nodes = self._get_replica_nodes(primary_fog_idx, num_replicas - 1, s_idx, rng=chrom_rng)
                for rep_fog_idx in replica_nodes:
                    placement.append({
                        "module_name": mod["module_name"],
                        "app": mod["app_id"],
                        "id_resource": self.system.fog_nodes[rep_fog_idx]
                    })
        
        return {"initialAllocation": placement}
    
    def _get_replica_nodes(self, primary_node_idx: int, num_replicas: int, service_idx: int,
                           rng: Optional[random.Random] = None) -> List[int]:
        """
        Select nodes for replicas using nearest-neighbor strategy.
        Prioritizes topologically close nodes with available resources.
        """
        if num_replicas <= 0:
            return []
        
        replica_nodes = []
        candidates = list(range(len(self.system.fog_nodes)))
        candidates.remove(primary_node_idx)  # Exclude primary node
        
        # Service resource needed for capacity checks
        service_resource = self.system.service_resources[service_idx]

        # If an RNG is provided, use it to randomly choose feasible nodes.
        # This makes replica placement stochastic/deterministic based on the RNG.
        if rng is not None:
            feasible = [c for c in candidates if self.system.fog_resources[c] >= service_resource]
            if len(feasible) >= num_replicas:
                return rng.sample(feasible, num_replicas)
            # Not enough feasible nodes: take all feasible, then randomly fill from remaining
            selected = feasible[:]
            remaining = [c for c in candidates if c not in selected]
            rng.shuffle(remaining)
            for c in remaining:
                if len(selected) >= num_replicas:
                    break
                selected.append(c)
            return selected[:num_replicas]

        # Default behavior: nearest-neighbor (topology-aware) selection
        # Sort candidates by distance from primary node if topology available
        if self.system.G is not None:
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
        for cand_idx in candidates:
            if len(replica_nodes) >= num_replicas:
                break
            if self.system.fog_resources[cand_idx] >= service_resource:
                replica_nodes.append(cand_idx)

        # If still not enough, fill from remaining candidates in order
        for cand_idx in candidates:
            if len(replica_nodes) >= num_replicas:
                break
            if cand_idx not in replica_nodes:
                replica_nodes.append(cand_idx)

        return replica_nodes[:num_replicas]
