from setuptools import setup, find_packages
from pathlib import Path

# Project root directory
here = Path(__file__).parent.resolve()

# Read dependencies from requirements.txt
with open(here / 'requirements.txt', encoding='utf-8') as f:
    install_requires = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith('#')
    ]

setup(
    name='system-trader',
    version='0.1.0',
    description='Autonomous day trading system',
    author='Your Name',
    license='MIT',
    packages=find_packages(),
    install_requires=install_requires,
    entry_points={
        'console_scripts': [
            'system-trader=main:main',
        ],
    },
)