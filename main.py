import os
import time
import json
import random
import logging.config
import argparse

import networkx as nx
from pathlib import Path
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from src.config import Config
from src.system_model import SystemModel
from src.ga_core import GACore
from src.nsgaii import NSGAII

from yafs.core import Sim
from yafs.application import create_applications_from_json
from yafs.topology import Topology

from yafs.placement import JSONPlacement
from yafs.path_routing import DeviceSpeedAwareRouting
from yafs.distribution import deterministic_distribution


def generate_ga_placement(cfg: Config):
    sysm = SystemModel(cfg)
    sysm.load()
    core = GACore(sysm, cfg)
    algo = NSGAII(core)
    pareto = algo.evolve()

    best_idx = next(iter(pareto.fronts[0])) if pareto.fronts[0] else 0
    chrom = pareto.population[best_idx]
    placement_json = core.chromosome_to_placement_json(chrom)
    return placement_json


def main(stop_time, it, folder_results, placement_override=None):

    """
    TOPOLOGY
    """
    t = Topology()
    dataNetwork = json.load(open('scenarios/networkDefinition.json'))
    t.load(dataNetwork)
    # nx.write_gexf(t.G,path+"graph_main") # you can export the Graph in multiples format to view in tools like Gephi, and so on.
    # t = loadTopology(path + 'test_GLP.gml')

    """
    APPLICATION or SERVICES
    """
    dataApp = json.load(open('scenarios/appDefinition.json'))
    apps = create_applications_from_json(dataApp)

    """
    SERVICE PLACEMENT: In order to run GAs, this *must* be replaced by a GA-based placement algorithm.
    The JSONPlacement reads a JSON file that indicates the initial placement of modules in the topology.
    """
    if placement_override is None:
        placementJson = json.load(open('scenarios/allocDefinition.json'))
    else:
        placementJson = placement_override
    placement = JSONPlacement(name="Placement", json=placementJson)

    """
    Defining ROUTING algorithm to define how path messages in the topology among modules
    """
    selectorPath = DeviceSpeedAwareRouting()

    """
    SIMULATION ENGINE
    """
    s = Sim(t, default_results_path=folder_results+"sim_trace")

    """
    Deploy services == APP's modules
    """
    for aName in apps.keys():
        s.deploy_app(apps[aName], placement, selectorPath) # Note: each app can have a different routing algorithm

    """
    Deploy users
    """
    userJSON = json.load(open('scenarios/usersDefinition.json'))
    for user in userJSON["sources"]:
        app_name = user["app"]
        app = s.apps[app_name]
        msg = app.get_message(user["message"])
        node = user["id_resource"]
        dist = deterministic_distribution(100, name="Deterministic")
        idDES = s.deploy_source(app_name, id_node=node, msg=msg, distribution=dist)

    """
    RUNNING - last step
    """
    logging.info(" Performing simulation: %i " % it)
    s.run(stop_time)  # To test deployments put test_initial_deploy a TRUE
    s.print_debug_assignaments()

    # Analysing the results. 
    dfl = pd.read_csv(folder_results+"sim_trace"+"_link.csv")
    print("Number of total messages between nodes: %i"%len(dfl))

    df = pd.read_csv(folder_results+"sim_trace.csv")
    print("Number of requests handled by deployed services: %i"%len(df))

    if not df.empty:
        dfapp2 = df[df.app == 2].copy()
        if not dfapp2.empty:
            dfapp2.loc[:,"transmission_time"] = dfapp2.time_emit - dfapp2.time_reception # Transmission time
            dfapp2.loc[:,"service_time"] = dfapp2.time_out - dfapp2.time_in

            print("The average service time of App2 is: %0.3f "%dfapp2["service_time"].mean())
            print("The App2 is deployed in the following nodes: %s"%np.unique(dfapp2["TOPO.dst"]))
            print("The number of instances of App2 deployed is: %s"%np.unique(dfapp2["DES.dst"]))



# -----------------------
# PLAY WITH THIS EXAMPLE!
# -----------------------
# Add another app2-instance in allocDefinition.json file adding the next data and run the main.py file again to see the new results:
# {
#   "module_name": "2_01",
#   "app": 2,
#   "id_resource": 3
# },
## What has happened to the results? Take a look at the network image available in the results folder to understand the "allocation" of app2-related entities.
    
# ! IMPORTANT. The scheduler & routing algorithm (aka. selectorPath = DeviceSpeedAwareRouting()) chooses the instance that will attend the request according to the latency -in this case-.
#  For that reason, the initial instance deployed at node 0 is not used. It is further away than the instance located at node3.
# Add another app2-user at node 16, add the next json inside of userDefinition.json file and try again. Enjoy it! 
# {
#   "id_resource": 16,
#   "app": 2,
#   "message": "M.USER.APP.2",
#   "lambda": 100
# },


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run YAFS simulation with optional GA-based placement")
    parser.add_argument("--use-ga", action="store_true", help="Generate placement via NSGA-II instead of scenarios/allocDefinition.json")
    parser.add_argument("--stop-time", type=int, default=20000, help="Simulation stop time")
    parser.add_argument("--iterations", type=int, default=1, help="Number of iterations")
    parser.add_argument("--sim-seed", type=int, default=0, help="Base seed for simulation-level randomness")
    parser.add_argument("--ga-model-seed", type=int, default=50, help="Seed for GA model aspects")
    parser.add_argument("--ga-population-seed", type=int, default=100, help="Seed for GA population init")
    parser.add_argument("--ga-evolution-seed", type=int, default=888, help="Seed for GA evolution ops")
    parser.add_argument("--ga-population-size", type=int, default=50, help="GA population size")
    parser.add_argument("--ga-generations", type=int, default=20, help="GA number of generations")
    parser.add_argument("--ga-mutation-probability", type=float, default=0.2, help="GA mutation probability")
    args = parser.parse_args()

    LOGGING_CONFIG = Path(__file__).parent / 'logging.ini'
    logging.config.fileConfig(LOGGING_CONFIG)

    folder_results = Path("results/")
    folder_results.mkdir(parents=True, exist_ok=True)
    folder_results = str(folder_results)+"/"

    cfg = Config(
        population_size=args.ga_population_size,
        mutation_probability=args.ga_mutation_probability,
        number_generations=args.ga_generations,
        model_seed=args.ga_model_seed,
        population_seed=args.ga_population_seed,
        evolution_seed=args.ga_evolution_seed,
    )

    placement_override = generate_ga_placement(cfg) if args.use_ga else None

    # Iteration for each experiment changing the seed of randoms
    for iteration in range(args.iterations):
        random.seed(args.sim_seed + iteration)
        logging.info("Running experiment it: - %i" % iteration)

        start_time = time.time()
        main(stop_time=args.stop_time,
             it=iteration, folder_results=folder_results, placement_override=placement_override)

        print("\n--- %s seconds ---" % (time.time() - start_time))

    print("Simulation Done!")


