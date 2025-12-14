# Experimenting with YAFS and Genetic Algorithms

YAFS (Yet Another Fog Simulator) and (some) GAs will be used to test and optimize the allocation of the applications.

## Installation

For systems that allow pip packages to be installed directly:

```bash
pip install -r requirements.txt
```

If not, `envsetup.sh` is provided for convenience. It makes a virtual environment and installs the dependencies if the current folder does not have one, and activates it.

## Running

Default (uses scenarios/allocDefinition.json):

```bash
python main.py
```

Run with GA-generated placement (still deterministic unless you change seeds):

```bash
python main.py --use-ga
```

Useful flags:
- `--stop-time` (default 20000)
- `--iterations` (default 1)
- `--sim-seed` (base seed added to iteration index)
- GA tuning: `--ga-model-seed`, `--ga-population-seed`, `--ga-evolution-seed`, `--ga-population-size`, `--ga-generations`, `--ga-mutation-probability`

Change seeds or GA parameters to introduce variability across runs; keep them fixed for reproducible traces.

## To-do

- Implement GAs (NSGA-II, NSGA-III and rule-based heuristics if possible)

### NSGA-II (fog-only) module
- New files under src/: config.py, system_model.py, population.py, ga_core.py, nsgaii.py
- Usage example to produce a placement JSON (to later plug into main):

```python
from src.config import Config
from src.system_model import SystemModel
from src.ga_core import GACore
from src.nsgaii import NSGAII

cfg = Config()
sysm = SystemModel(cfg)
sysm.load()
core = GACore(sysm, cfg)
algo = NSGAII(core)
pareto = algo.evolve()

# Take first solution and export placement-like JSON
best_idx = next(iter(pareto.fronts[0])) if pareto.fronts[0] else 0
chrom = pareto.population[best_idx]
placement_json = core.chromosome_to_placement_json(chrom)
print(placement_json)
```

#### CLI Usage

Generate GA placement with custom parameters:

```bash
# Basic usage
python main.py --use-ga

# With custom GA parameters
python main.py --use-ga --ga-population-size 30 --ga-generations 50 --ga-mutation-probability 0.3

# Control randomness via seeds (deterministic if seeds fixed, variable if different)
python main.py --use-ga --ga-model-seed 42 --ga-population-seed 42 --ga-evolution-seed 42
```

You can then pass the generated placement JSON to YAFS's JSONPlacement instead of loading from scenarios/allocDefinition.json.

#### Enhancements

See [ENHANCEMENTS.md](ENHANCEMENTS.md) for detailed information about advanced features and alignment with the original research paper.

## License

This project is licensed under the GPLv3 License - see the LICENSE file for details.

I use YAFS (YAFS3 branch), which is licensed under the MIT License - see the [YAFS](https://github.com/acsicuib/YAFS) repository for details.

This project uses code based from GA4FogPlacement, which is licensed under the GPLv3 License - see the [GA4FogPlacement](https://github.com/acsicuib/GA4FogPlacement) repository for details.