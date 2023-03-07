from setuptools import setup
from setuptools import find_packages

setup(name='GATAERP',
      description='Tool for relationship prediction of cancer patients',
      author='hengyuan zhang',
      install_requires=['networkx',
                        'numpy',
                        'scikit-learn',
                        'xgboost',
                        'scipy',  # pip install
                        'pandas',
                        'rpy2',  # pip install rpy2
                        'torch',  # conda install pytorch pytorch-cuda=11.6 -c pytorch -c nvidia
                        'matplotlib',  # 3.7 has bugs , conda install matplotlib==3.5.1
                        'seaborn',
                        'requests'
                        ],
      packages=find_packages())
