from dataclasses import dataclass


@dataclass
class Config:
    # Basic GA parameters
    population_size: int = 50
    mutation_probability: float = 0.2
    number_generations: int = 20
    max_replicas: int = 3  # Maximum replicas per service (1 = no replicas, 2 = 1 additional, etc.)

    # Seeds
    model_seed: int = 50
    population_seed: int = 100
    evolution_seed: int = 888

    # Scenario files (reuse existing ones in scenarios/)
    topology_json: str = "scenarios/networkDefinition.json"
    application_json: str = "scenarios/appDefinition.json"
    user_json: str = "scenarios/usersDefinition.json"
