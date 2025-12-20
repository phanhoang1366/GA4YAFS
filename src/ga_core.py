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
        
        # BIẾN ĐỘNG CHO ADAPTIVE MUTATION
        # Khởi tạo bằng giá trị start từ config
        self.current_mutation_prob = cfg.mutation_prob_start 

    # --- TÍCH HỢP HEURISTIC INITIALIZATION ---
    def generate_population(self, pop: Population):
        pop.population = []
        
        # 1. Tính số lượng cá thể Heuristic
        num_heuristic = int(self.population_size * self.cfg.heuristic_ratio)
        
        # Chuẩn bị danh sách các node mạnh nhất (sắp xếp giảm dần theo Resource)
        # self.system.fog_resources chứa capacity của từng node
        fog_res = np.array(self.system.fog_resources)
        # Lấy index của các node, sort giảm dần theo tài nguyên
        sorted_node_indices = np.argsort(fog_res)[::-1] 
        
        # Chọn top 20% node mạnh nhất để làm ứng viên cho heuristic
        top_n_count = max(1, int(len(self.system.fog_nodes) * 0.2))
        top_nodes = sorted_node_indices[:top_n_count]

        for i in range(self.population_size):
            chrom = []
            
            # Nếu nằm trong số lượng heuristic -> Dùng chiến thuật tham lam (Greedy)
            if i < num_heuristic:
                # Logic Heuristic: Gán service vào các node mạnh nhất
                # Để tránh tất cả giống hệt nhau, ta chọn ngẫu nhiên trong nhóm Top Node
                for _ in range(self.system.service_number):
                    # Chọn 1 node bất kỳ trong nhóm top node mạnh nhất
                    chosen_node = self.rnd_pop.choice(top_nodes)
                    chrom.append(chosen_node)
            else:
                # Logic Random truyền thống (cho phần còn lại của dân số)
                chrom = [self.rnd_pop.randint(0, len(self.system.fog_nodes) - 1)
                         for _ in range(self.system.service_number)]
            
            pop.population.append(chrom)

        # Initialize metadata (giữ nguyên code cũ)
        pop.fitness = [{} for _ in range(len(pop.population))]
        pop.dominates_to = [set() for _ in range(len(pop.population))]
        pop.dominated_by = [set() for _ in range(len(pop.population))]
        pop.fronts = [set() for _ in range(len(pop.population))]
        pop.crowding_distances = [0.0 for _ in range(len(pop.population))]
        self.calculate_population_fitness_objectives(pop)

    def mutate(self, chrom: List[int]) -> List[int]:
        """
        Apply mutation using the ADAPTIVE probability.
        """
        mutation_type = self.rnd_evol.choice(['random', 'shuffle', 'neighbor'])
        newc = chrom[:]
        
        # LƯU Ý: Dùng self.current_mutation_prob thay vì cfg cố định
        prob = self.current_mutation_prob 
        
        if mutation_type == 'random':
            for i in range(len(newc)):
                if self.rnd_evol.random() < prob:
                    newc[i] = self.rnd_evol.randint(0, len(self.system.fog_nodes) - 1)
        
        elif mutation_type == 'shuffle':
            if len(newc) > 1:
                # Số lần swap cũng phụ thuộc vào prob
                num_swaps = max(1, int(prob * len(newc)))
                for _ in range(num_swaps):
                    i = self.rnd_evol.randint(0, len(newc) - 1)
                    j = self.rnd_evol.randint(0, len(newc) - 1)
                    newc[i], newc[j] = newc[j], newc[i]
        
        elif mutation_type == 'neighbor':
            if self.system.G is None:
                for i in range(len(newc)):
                    if self.rnd_evol.random() < prob:
                        newc[i] = self.rnd_evol.randint(0, len(self.system.fog_nodes) - 1)
            else:
                for i in range(len(newc)):
                    if self.rnd_evol.random() < prob:
                        current_node = newc[i]
                        neighbors = list(self.system.G.neighbors(self.system.fog_nodes[current_node]))
                        if neighbors:
                            neighbor_indices = [self.system.fog_nodes.index(n) for n in neighbors if n in self.system.fog_nodes]
                            if neighbor_indices:
                                newc[i] = self.rnd_evol.choice(neighbor_indices)
                            else:
                                newc[i] = self.rnd_evol.randint(0, len(self.system.fog_nodes) - 1)
                        else:
                            newc[i] = self.rnd_evol.randint(0, len(self.system.fog_nodes) - 1)
        
        return newc
        
    # ... (Giữ nguyên phần crossover và calculate_fitness, không thay đổi) ...
    def crossover(self, a: List[int], b: List[int]) -> List[int]:
        # [Copy paste lại nội dung hàm crossover cũ vào đây hoặc giữ nguyên file nếu chỉ sửa phần trên]
        # (Để tiết kiệm dòng hiển thị, tôi giả định bạn giữ nguyên phần crossover cũ)
        la = len(a)
        lb = len(b)
        if la == 0 and lb == 0: return []
        if la != lb:
            m = min(la, lb)
            child = (a[:m] if self.rnd_evol.random() < 0.5 else b[:m])
            base = a if len(a) >= len(b) else b
            child += base[m:]
            return child
        n = la
        if n <= 2:
            return [a[i] if self.rnd_evol.random() < 0.5 else b[i] for i in range(n)]

        groups = []
        try:
            mods = getattr(self.system, 'service_modules', [])
            if mods and len(mods) == n:
                current_app = mods[0]['app_id']
                start_idx = 0
                for i in range(1, n):
                    if mods[i]['app_id'] != current_app:
                        groups.append((start_idx, i))
                        start_idx = i
                        current_app = mods[i]['app_id']
                groups.append((start_idx, n))
        except Exception:
            groups = []

        child: List[int] = []
        if groups and len(groups) > 0:
            take_from_a = self.rnd_evol.random() < 0.5
            for (g_start, g_end) in groups:
                g_len = g_end - g_start
                if g_len <= 1:
                    gene = a[g_start] if take_from_a else b[g_start]
                    child.append(gene)
                else:
                    cut = self.rnd_evol.randint(g_start + 1, g_end - 1)
                    seg1 = (a[g_start:cut] if take_from_a else b[g_start:cut])
                    seg2 = (b[cut:g_end] if take_from_a else a[cut:g_end])
                    child.extend(seg1)
                    child.extend(seg2)
                take_from_a = not take_from_a
            return child
        else:
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
        for idx, chrom in enumerate(pop.population):
            latency = self._calculate_latency(chrom)
            spread = self._calculate_spread(chrom)
            utilization = self._calculate_resource_utilization(chrom)
            underutilization = 1.0 - utilization
            pop.fitness[idx] = {
                "latency": latency,
                "spread": spread,
                "underutilization": underutilization,
                "index": idx
            }

    def _calculate_latency(self, chrom: List[int]) -> float:
        if len(chrom) == 0 or len(self.system.service_matrix) == 0: return 0.0
        total_latency = 0.0
        num_dependencies = 0
        for src_service_idx in range(len(chrom)):
            src_node_idx = chrom[src_service_idx]
            for dst_service_idx in range(len(chrom)):
                if self.system.service_matrix[src_service_idx][dst_service_idx] == 1:
                    dst_node_idx = chrom[dst_service_idx]
                    total_latency += self.system.dev_distance_matrix[src_node_idx][dst_node_idx]
                    num_dependencies += 1
        if num_dependencies > 0:
            return total_latency / (num_dependencies * self.system.average_path_length)
        return 0.0

    def _calculate_spread(self, chrom: List[int]) -> float:
        if len(chrom) == 0: return 0.0
        placement_count = [0 for _ in self.system.fog_nodes]
        for fog_idx in chrom: placement_count[fog_idx] += 1
        non_zero_counts = [c for c in placement_count if c > 0]
        if len(non_zero_counts) == 0: return 1.0
        mean_count = np.mean(non_zero_counts)
        std_count = np.std(non_zero_counts)
        count_variance = (std_count / mean_count) if mean_count > 0 else 0.0
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
                max_possible_dist = 0.0
                for row in self.system.dev_distance_matrix:
                    for val in row:
                        if val > max_possible_dist: max_possible_dist = val
                if max_possible_dist > 0:
                    normalized_mean_dist = mean_dist / max_possible_dist
                    distance_variance = 1.0 - normalized_mean_dist
                else: distance_variance = 0.0
        spread_score = 0.6 * count_variance + 0.4 * distance_variance
        return float(spread_score)

    def _calculate_resource_utilization(self, chrom: List[int]) -> float:
        # Ngưỡng để xác định Cloud (ví dụ: lớn hơn 10^12 là Cloud)
        CLOUD_THRESHOLD = 1e12
        
        total_used_fog = 0
        total_available_fog = 0
        
        # 1. Tính tổng tài nguyên khả dụng của các node KHÔNG PHẢI CLOUD
        # Đồng thời xác định index của các node này để lọc việc sử dụng sau đó
        valid_fog_indices = set()
        for i, cap in enumerate(self.system.fog_resources):
            if cap < CLOUD_THRESHOLD:
                total_available_fog += cap
                valid_fog_indices.add(i)
        
        # Nếu không có Fog node nào (chỉ có Cloud), trả về 0 hoặc xử lý riêng
        if total_available_fog == 0: 
            return 0.0

        # 2. Tính tài nguyên ĐANG ĐƯỢC SỬ DỤNG trên các node Fog này
        for s_idx, fog_idx in enumerate(chrom):
            # Chỉ cộng nếu service này được đặt lên Fog node (không phải Cloud)
            if fog_idx in valid_fog_indices:
                total_used_fog += self.system.service_resources[s_idx]

        # 3. Tính tỷ lệ sử dụng thực tế của hạ tầng Fog
        utilization = total_used_fog / total_available_fog
        
        return utilization

    def is_feasible(self, chrom: List[int]) -> bool:
        if len(chrom) != len(self.system.service_resources): return False
        used = [0 for _ in self.system.fog_nodes]
        for s_idx, fog_idx in enumerate(chrom):
            if fog_idx < 0 or fog_idx >= len(self.system.fog_nodes): return False
            used[fog_idx] += self.system.service_resources[s_idx]
        for node_idx, resources_used in enumerate(used):
            if resources_used > self.system.fog_resources[node_idx]: return False
        return True

    def chromosome_to_placement_json(self, chrom: List[int]) -> Dict[str, Any]:
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