# Matrix Approximation Project

This project compares different matrix approximation methods.

## Files

- `approximation.py`: contains the approximation algorithms.
- `data.py`: loads data and creates kernel matrices.
- `synthetic.py`: creates artificial test matrices.
- `error_analysis.py`: computes errors and execution time.
- `tests.py`: runs the experiments and saves the plots.

## Requirements

Install the needed libraries:

```bash
pip install numpy matplotlib scikit-learn
```

## How to run

Run:

```bash
python tests.py
```

The plots will be saved in the folder:

```bash
plots/
```

## What the project does

The project tests matrix approximation methods such as:

- fpCA
- ppCA
- random ppCA
- weighted ppCA
- adaptive ppCA
- SVD for comparison

It compares them using the relative Frobenius error.

## Output

The program creates PDF plots showing how the error changes with the rank.

## Notes

Some tests may take time because the matrices can be large.
