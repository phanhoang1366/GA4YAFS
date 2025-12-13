import json
import random
from typing import List, Dict, Any

import networkx as nx

from .config import Config


class SystemModel:
    """
    Minimal fog-only system model:
    - Loads topology from scenarios/networkDefinition.json
    - Treats every node as a fog node (no clusters/gateways/cloud)
    - Loads services (apps) and users, exposing resources and distances
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rnd = random.Random(cfg.model_seed)

        self.G = None
        self.fog_nodes: List[int] = []
        self.fog_resources: List[int] = []
        self.fog_speed_cpu: List[int] = []

        self.service_resources: List[int] = []
        self.service_number: int = 0
        self.service_modules: List[dict] = []  # [{app_id:int, module_name:str, resource:int}]

        self.dev_distance_matrix: List[List[float]] = []

    def load(self):
        # Topology: load and mark all nodes as fog
        with open(self.cfg.topology_json, "r") as f:
            topo = json.load(f)

        # Support both YAFS docs keys (entity/link) and earlier (nodes/links)
        entities = topo.get("entity") or topo.get("nodes") or []
        links = topo.get("link") or topo.get("links") or []

        G = nx.Graph()
        for n in entities:
            node_id = n.get("id")
            # RAM is the documented memory field; fall back to capacity if present
            res = n.get("RAM", n.get("capacity", 10))
            # IPT is the documented processing rate; fall back to speed
            speed = n.get("IPT", n.get("speed", 100))
            G.add_node(node_id, nodetype="fog", capacity=res, speed=speed)

        for l in links:
            src = l.get("s", l.get("source"))
            dst = l.get("d", l.get("target"))
            bw = l.get("BW", l.get("bandwidth", 1))
            lat = l.get("PR", l.get("latency", 1))
            G.add_edge(src, dst, weight=lat, bandwidth=bw, latency=lat)

        self.G = G
        self.fog_nodes = list(G.nodes())
        self.fog_resources = [G.nodes[i].get("capacity", 10) for i in self.fog_nodes]
        self.fog_speed_cpu = [G.nodes[i].get("speed", 100) for i in self.fog_nodes]

        # Distances
        n = len(self.fog_nodes)
        self.dev_distance_matrix = [[0.0 for _ in range(n)] for _ in range(n)]
        index_of = {node: idx for idx, node in enumerate(self.fog_nodes)}
        for u in self.fog_nodes:
            dist = nx.single_source_dijkstra_path_length(G, u, weight="latency")
            for v, d in dist.items():
                self.dev_distance_matrix[index_of[u]][index_of[v]] = float(d)

        # Applications (services): collect modules per app with their resource proxy
        with open(self.cfg.application_json, "r") as f:
            apps_json = json.load(f)

        self.service_resources = []
        self.service_modules = []
        for app in apps_json:
            app_name = app.get("name", app.get("id", 0))
            app_id = str(app_name)
            modules = app.get("modules") or app.get("module") or []
            for m in modules:
                # Modules may come as dicts keyed by module name per docs, or as dicts with name/module_name
                if isinstance(m, dict) and len(m) == 1 and not m.get("name") and not m.get("module_name"):
                    mod_name, mod_def = next(iter(m.items()))
                    res = int(mod_def.get("RAM", 1)) if isinstance(mod_def, dict) else 1
                else:
                    mod_name = m.get("name") or m.get("module_name") or "module"
                    res = int(m.get("RAM", 1))
                self.service_modules.append({"app_id": app_id, "module_name": mod_name, "resource": max(1, res)})
                self.service_resources.append(max(1, res))

        self.service_number = len(self.service_resources)
