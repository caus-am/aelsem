# aelsem
Python functions for Algebraic Equivalence of Linear Structural Equation Models

## Files
  - aelsem.py contains many useful functions for dealing with linear
    structural equation models, including:
      - maximum likelihood fitting with RICF;
      - model selection;
      - grouping models based on theoretical or empirical results;
      - outputting (sets of) mixed graphs to .tex format.

  - aelsem_ystruct.py performs a model selection experiment,
    demonstrating how some of the functions in aelsem.py might be
    used.

  - ystruct/* generates the data for the above experiment (originally
    analysed by Joris Mooij and Jerome Cremers (2015)). The files in
    ystruct/dai/ come from libDAI:
    https://bitbucket.org/jorism/libdai.git


## Paper
The theoretical results that this software relies on, as well as the
graph pattern notation for succinctly representing sets of mixed
graphs, are described in the following paper:

> Thijs van Ommen and Joris M. Mooij, Algebraic Equivalence of Linear
> Structural Equation Models, Proceedings of the 33rd Annual
> Conference on Uncertainty in Artificial Intelligence (UAI-17), 2017.

If you use this software for a scientific publication, we would
appreciate it if you could cite our paper.


## Example usage
To get a table of all 4-node acyclic algebraic equivalence classes (as
in Appendix B of the UAI 2017 paper):

```
python aelsem.py
cd tex
pdflatex table.tex
```

To repeat the experiments from the paper (just for p=10):

```
  cd ystruct
  ./build.sh
  cd experiments_uai2015
  ./run_experiment_p.sh 10
  cd ../..
  python aelsem_ystruct.py p10_final
```
