from dataclasses import dataclass

@dataclass
class Config:
    # Basic GA parameters
    population_size: int = 100       # file csv
    number_generations: int = 100    
    
    # --- ADAPTIVE MUTATION PARAMETERS ---
    mutation_prob_start: float = 0.5   
    mutation_prob_end: float = 0.01    
    
    # --- HEURISTIC INITIALIZATION PARAMETERS ---
    heuristic_ratio: float = 0.2 

    # Seeds
    model_seed: int = 50
    population_seed: int = 100
    evolution_seed: int = 888

    # Scenario files
    topology_json: str = "scenarios/networkDefinition.json"
    application_json: str = "scenarios/appDefinition.json"
    user_json: str = "scenarios/usersDefinition.json"