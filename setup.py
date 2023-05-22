from setuptools import setup, find_packages

setup(
    name='factorlib',
    version='0.0.1',
    packages=find_packages(),
    url='',
    license='',
    author='jakezur',
    author_email='jakezur@umich.edu',
    description='Streamlines experimental pipeline for testing and creating factor models',
    install_requires=['pandas_datareader',
                      'tqdm',
                      'pandas',
                      'numpy==1.23.1',
                      'yfinance',
                      'matplotlib',
                      'scikit-learn',
                      'statsmodels',
                      'quantstats',
                      'getFamaFrenchFactors',
                      'scipy',
                      'sklearn',
                      'xgboost',
                      'pykalman',
                      'PyWavelets',
                      'pyarrow',
                      'cloudpickle',
                      'prettytable',
                      'ipython',
                      'Jupyter',
                      'fastai',
                      'pandas_ta',
                      'pandarallel',
                      'shap'],
)

