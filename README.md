[![Project generated with PyScaffold](https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold)](https://pyscaffold.org/)
<!-- These are examples of badges you might also want to add to your README. Update the URLs accordingly.
[![Built Status](https://api.cirrus-ci.com/github/<USER>/ssl_wafermap.svg?branch=main)](https://cirrus-ci.com/github/<USER>/ssl_wafermap)
[![ReadTheDocs](https://readthedocs.org/projects/ssl_wafermap/badge/?version=latest)](https://ssl_wafermap.readthedocs.io/en/stable/)
[![Coveralls](https://img.shields.io/coveralls/github/<USER>/ssl_wafermap/main.svg)](https://coveralls.io/r/<USER>/ssl_wafermap)
[![PyPI-Server](https://img.shields.io/pypi/v/ssl_wafermap.svg)](https://pypi.org/project/ssl_wafermap/)
[![Conda-Forge](https://img.shields.io/conda/vn/conda-forge/ssl_wafermap.svg)](https://anaconda.org/conda-forge/ssl_wafermap)
[![Monthly Downloads](https://pepy.tech/badge/ssl_wafermap/month)](https://pepy.tech/project/ssl_wafermap)
[![Twitter](https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter)](https://twitter.com/ssl_wafermap)
-->

# ssl_wafermap

<p align="center"><em>How much can self-supervised models learn about semiconductor data without being told what to look for?</em></p>
<p align="center">
  <img src="https://user-images.githubusercontent.com/68213622/230577877-591e7ae2-bdf7-4fee-9333-90d55822ce7a.png" alt="watercolor integrated circuits" />
</p>


Self-supervised learning is responsible for many recent breakthroughs in machine learning. Nearly every major development in NLP relies on self-supervised pretraining to some extent, and in vision, techniques like contrastive learning are gaining popularity as well. On ImageNet style data ([and even non-curated datasets of natural images](https://github.com/facebookresearch/vissl/blob/66a1f1997d2135f90a429ec3a37a4a503869f2a9/projects/SEER/README.md)), self-supervised learning has been shown to be a powerful tool for learning useful representations of images without the need for large amounts of labeled data.

It is often unclear, however, how effectively these techniques can be applied to images in other domains. How well can self-supervised learning be used in semiconductor manufacturing, where data is large, unlabeled, and highly imbalanced? This work provides thorough evaluations of some of the most popular self-supervised learning techniques on semiconductor wafer map data. Everything is done using [lightly](https://github.com/lightly-ai/lightly), an excellent package for self-supervised learning on images using PyTorch. More than a dozen joint embedding and masked image modeling frameworks are put to the test on the WM-811K and MixedWM38 datasets.

## Installation

In order to set up the necessary environment:

1. review and uncomment what you need in `environment.yml` and create an environment `ssl_wafermap` with the help of [conda]:
   ```
   conda env create -f environment.yml
   ```
2. activate the new environment with:
   ```
   conda activate ssl_wafermap
   ```

> **_NOTE:_**  The conda environment will have ssl_wafermap installed in editable mode.
> Some changes, e.g. in `setup.cfg`, might require you to run `pip install -e .` again.


Optional and needed only once after `git clone`:

3. install several [pre-commit] git hooks with:
   ```bash
   pre-commit install
   # You might also want to run `pre-commit autoupdate`
   ```
   and checkout the configuration under `.pre-commit-config.yaml`.
   The `-n, --no-verify` flag of `git commit` can be used to deactivate pre-commit hooks temporarily.

4. install [nbstripout] git hooks to remove the output cells of committed notebooks with:
   ```bash
   nbstripout --install --attributes notebooks/.gitattributes
   ```
   This is useful to avoid large diffs due to plots in your notebooks.
   A simple `nbstripout --uninstall` will revert these changes.


Then take a look into the `scripts` and `notebooks` folders.

## Dependency Management & Reproducibility

1. Always keep your abstract (unpinned) dependencies updated in `environment.yml` and eventually
   in `setup.cfg` if you want to ship and install your package via `pip` later on.
2. Create concrete dependencies as `environment.lock.yml` for the exact reproduction of your
   environment with:
   ```bash
   conda env export -n ssl_wafermap -f environment.lock.yml
   ```
   For multi-OS development, consider using `--no-builds` during the export.
3. Update your current environment with respect to a new `environment.lock.yml` using:
   ```bash
   conda env update -f environment.lock.yml --prune
   ```
## Project Organization

```
├── AUTHORS.md              <- List of developers and maintainers.
├── CHANGELOG.md            <- Changelog to keep track of new features and fixes.
├── CONTRIBUTING.md         <- Guidelines for contributing to this project.
├── Dockerfile              <- Build a docker container with `docker build .`.
├── LICENSE.txt             <- License as chosen on the command-line.
├── README.md               <- The top-level README for developers.
├── configs                 <- Directory for configurations of model & application.
├── data
│   ├── external            <- Data from third party sources.
│   ├── interim             <- Intermediate data that has been transformed.
│   ├── processed           <- The final, canonical data sets for modeling.
│   └── raw                 <- The original, immutable data dump.
├── docs                    <- Directory for Sphinx documentation in rst or md.
├── environment.yml         <- The conda environment file for reproducibility.
├── models                  <- Trained and serialized models, model predictions,
│                              or model summaries.
├── notebooks               <- Jupyter notebooks. Naming convention is a number (for
│                              ordering), the creator's initials and a description,
│                              e.g. `1.0-fw-initial-data-exploration`.
├── pyproject.toml          <- Build configuration. Don't change! Use `pip install -e .`
│                              to install for development or to build `tox -e build`.
├── references              <- Data dictionaries, manuals, and all other materials.
├── reports                 <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures             <- Generated plots and figures for reports.
├── scripts                 <- Analysis and production scripts which import the
│                              actual PYTHON_PKG, e.g. train_model.
├── setup.cfg               <- Declarative configuration of your project.
├── setup.py                <- [DEPRECATED] Use `python setup.py develop` to install for
│                              development or `python setup.py bdist_wheel` to build.
├── src
│   └── ssl_wafermap        <- Actual Python package where the main functionality goes.
├── tests                   <- Unit tests which can be run with `pytest`.
├── .coveragerc             <- Configuration for coverage reports of unit tests.
├── .isort.cfg              <- Configuration for git hook that sorts imports.
└── .pre-commit-config.yaml <- Configuration of pre-commit git hooks.
```

<!-- pyscaffold-notes -->

## Note

This project has been set up using [PyScaffold] 4.1.4 and the [dsproject extension] 0.7.2.

[conda]: https://docs.conda.io/
[pre-commit]: https://pre-commit.com/
[Jupyter]: https://jupyter.org/
[nbstripout]: https://github.com/kynan/nbstripout
[Google style]: http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
[PyScaffold]: https://pyscaffold.org/
[dsproject extension]: https://github.com/pyscaffold/pyscaffoldext-dsproject
