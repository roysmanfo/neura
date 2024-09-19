from setuptools import setup, find_packages
from pathlib import Path

DIR = Path(__file__).parent

with open(str(DIR / 'requirements.txt'), 'r') as fp:
    lines = fp.read().splitlines()
    REQUIREMENTS = lines if lines else []

setup(
    name='nncl',
    version="0.1",
    author='roysmanfo',
    url='https://github.com/roysmanfo/neura',
    install_requires=REQUIREMENTS,
    packages=find_packages(include=['neura']),
)
