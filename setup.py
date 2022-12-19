""" setup.py - Main setup module """
import os
from setuptools import setup, find_packages

HERE = os.path.abspath(os.path.dirname(__file__))
README = open(os.path.join(HERE, 'README.md')).read()
VERSION = '2.0'

# Publicly Available Packages (PyPi)
INSTALL_REQUIRES = [
    'cython',
    'scikit-learn',
    'scipy',
    'numpy',
    'pytest',
    'mock',
    'black',
    'pylint',
    'docformatter',
    'transformers',
]

setup(
    name='QA_ZRE',
    version=VERSION,
    description="Codes developed during QA-ZRE project",
    long_description=README,
    classifiers=['Programming Language :: Python :: 3.7'],
    keywords="NLP, Machine Learning",
    author="Saeed Najafi",
    author_email="snajafi@ualberta.com",
    license='MIT',
    packages=find_packages(exclude=['docs', 'tests']),
    include_package_data=True,
    zip_safe=False,
    install_requires=INSTALL_REQUIRES
)
