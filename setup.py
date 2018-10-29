from setuptools import setup

setup(
    name='qcommunity',
    description=
    'Quantum Local Search framework for graph modularity optimization',
    author='Ruslan Shaydulin',
    author_email='rshaydu@g.clemson.edu',
    packages=['qcommunity'],
    install_requires=[
        'qiskit', 'networkx', 'numpy', 'matplotlib', 'joblib', 'pyomo',
        'progressbar2'
    ],
    zip_safe=False)
