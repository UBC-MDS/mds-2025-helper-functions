# mds_2025_helper_functions

A package to streamline common code chunks executed by students in the UBC MDS program circa 2025. 

## Functions
- compare_model_scores() - a function that takes multiple models and returns a table of mean CV scores for each for easy comparison.
- perform_eda() - a function to perform exploratory data analysis on a dataset
- [PLACEHOLDER - SAMKARLYGASH'S FUNCTION]
- [PLACEHOLDER - XI'S FUNCTION]

## Similar packages
- While this package extends cross-validation from [scikit-learn](https://scikit-learn.org/stable/), there are no known packages that provide CV score comparison similar to compare_model_scores(). The most similar is the summary_cv() function in the [CrossPy](https://github.com/UBC-MDS/CrossPy) package, which summarizes CV scores for a single model.

- While the `ProfileReport` class from the [ydata-profiling](https://github.com/ydataai/ydata-profiling) package provides automated exploratory data analysis and reporting, there are no known packages that offer the same level of flexible, on-demand visualizations and insights as the `perform_eda()` function. The most similar functionality is available in pandas-profiling, which generates detailed HTML reports but lacks the modular, interactive approach that `perform_eda()` provides for tailoring EDA to specific datasets and workflows.

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
