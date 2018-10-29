### Reproducing the results of "Community Detection in Networks On Small Quantum Computers"

To reproduce the results, run `single_level_refinement.py` on graphs in `data/graphs`.

Seeds used to obtain the results in the paper are in `data/pmes2018/seeds.txt`

Example for QAOA:

```
for g in data/graphs/*/out.*;
do
    ./single_level_refinement.py --graph  --method qaoa --label ibm_16_0926 --iter-size 16 --verbose --stopping-criteria 3 --seed 1 --subset top_gain --backend IBMQX --backend-device ibmq_16_melbourne
done

./single_level_refinement.py --pajek data/graphs/random_modular_graph_2000_12_2_q_0.45.p --method qaoa --label ibm_16_0926 --iter-size 16 --verbose --stopping-criteria 3 --seed 1 --subset top_gain --backend IBMQX --backend-device ibmq_16_melbourne
```

Use `./single_level_refinement -h` for all available options. Gurobi local solver can be used by passign `--method optimal`. Gurobi solver uses temporary files and requires environmental variable `TMPDIR` to be defined (e.g. `export TMPDIR=/tmp`)

D-Wave backend is available on request. Contact us directly at rshaydu@g.clemson.edu if you want to use our D-Wave backend.
