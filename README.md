# Experimenting with YAFS and Genetic Algorithms

YAFS (Yet Another Fog Simulator) and (some) GAs will be used to test and optimize the allocation of the applications.

## Installation

For systems that allow pip packages to be installed directly:

```bash
pip install -r requirements.txt
```

If not, `envsetup.sh` is provided for convenience. It makes a virtual environment and installs the dependencies if the current folder does not have one, and activates it.

## Running

```bash
python main.py
```

## To-do

- Implement GAs (NSGA-II, NSGA-III and rule-based heuristics if possible)