# QCommunity

Set of tools that detect communities by optimizing modularity using QAOA.

### Dependencies

##### Basic usage

Gurobi local solver requires a Gurobi installation. On most HPC systems `module load gurobi` or equivalent suffices.

```
conda config --add channels intel # Recommended
conda create --name qcommunity python=3.7
source activate qcommunity
git clone git@github.com:rsln-s/ibmqxbackend.git
cd ibmqxbackend
pip install -e .
cd ..
git clone git@github.com:rsln-s/QCommunity.git
cd QCommunity
pip install -e .
```

### Reproducing the results of "Community Detection in Networks On Small Quantum Computers"

See `qcommunity/modularity/REPRODUCE.md`

