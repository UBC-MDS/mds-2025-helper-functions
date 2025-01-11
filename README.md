# mds_2025_helper_functions

A package to streamline common code chunks executed by students in the UBC MDS program circa 2025. 

## Functions
- compare_model_scores() - a function that takes multiple models and returns a table of mean CV scores for each for easy comparison.
- [PLACEHOLDER - SAM'S FUNCTION]
- [PLACEHOLDER - SAMKARLYGASH'S FUNCTION]
- htv() - (Hypothesis Test Visualization) provide good plots for user's hypothesis test result, easier to understand what happend in test rather than just number.

## Similar packages
While this package extends cross-validation from [scikit-learn](https://scikit-learn.org/stable/), there are no known packages that provide CV score comparison similar to compare_model_scores(). The most similar is the summary_cv() function in the [CrossPy](https://github.com/UBC-MDS/CrossPy) package, which summarizes CV scores for a single model.

There is no similar function could provide plot for hypothesis test output. Data Scientist do it manually, but it is not friendly for learner.
## Installation

```bash
$ pip install mds_2025_helper_functions
```

## Usage

- TODO

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## Contributors

Karlygash Zhakupbayeva, Samuel Adetsi, Xi Cu, Michael Hewlett

## License

`mds_2025_helper_functions` was created by Karlygash Zhakupbayeva, Samuel Adetsi, Xi Cu, Michael Hewlett. It is licensed under the terms of the MIT license.

## Credits

`mds_2025_helper_functions` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
