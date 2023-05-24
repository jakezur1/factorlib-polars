# Factorlib
Factorlib is a python package (pip installation coming soon) that provides a quick and intuitive system to create, 
backtest, and evaluate alpha factor models. Factorlib was originally created to be used in the context of the stock
market, but it can be applied to a variety of time series prediction problems, like horse racing or sports betting.

[Installation](#installation) •
[Documentation](#documentation) •
[Examples](#examples) •

## Features

- Provide an interface to create factors from datasets or files.

- Allows users to quickly add factors to a factor model.

- Utilizes `polars`, the lightning-fast dataframe library, to manage factors and data.

- Extremely fast and memory efficient.

- Supports walk-forward-optimization for backtesting models, with hyperparameter and tuning options available.

- Includes an entire Statistics module for easy model evaluation after backtesting.

- Supports a wide array of transformations and helper utility functions for creating robust and unique factors.

## Table of Contents

- [Installation](#installation)
  - [Cloning the Repository](#cloning-the-repository)
  - [Usage Prerequisites](#usgae-prerequisites)
- [Documentation](#documentation)
  - [API documentation and tutorials](#api-documentation-and-tutorials)
- [Examples](#examples)

# Installation
<sup>[(Back to top)](#table-of-contents)</sup>
## Cloning the Repository
To use Factorlib, you can clone the repo directly from this page. Copy the http or ssh url from the green `Code` button
in the top right corner of the repo and follow the steps below.

Open your terminal and `cd` to the directory where you'd like to use Factorlib.
```shell
$ cd Documents/destination
```

Then, clone the repository by typing the following command.
```shell
$ git clone git@github.com:jakezur1/factorlib-polars.git
```

Once you see this message in your terminal, the repo has been successfully cloned to the current working directory of 
your terminal and can be used like any other module in python.
```shell
Cloning into 'factorlib-polars'...
remote: Enumerating objects: 108, done.
remote: Counting objects: 100% (108/108), done.
remote: Compressing objects: 100% (78/78), done.
remote: Total 108 (delta 38), reused 89 (delta 22), pack-reused 0
Receiving objects: 100% (108/108), 839.28 KiB | 4.51 MiB/s, done.
Resolving deltas: 100% (38/38), done.
```

## Usage Prerequisites
To use factorlib as a feature-engineer or developer for the creator of factorlib, you will need the data that the 
creator used. Please request this data from jakezur@umich.edu to get access to the drive. Once you have obtained the
data, follow these steps to ensure the correct file structure.

Open your terminal and `cd` into the factorlib repository.
```shell
cd Documents/destination/factorlib
```

Create a data folder named `data` in the root of factorlib.
```shell
mkdir data
```

Finally, upload the data to this folder from Google Drive, or, if you have already downloaded the data, drag and drop
the data from your downloads to the newly created `data` folder in the factorlib.

# Documentation
<sup>[(Back to top)](#table-of-contents)</sup>

COMING SOON

# Examples
COMING SOON
