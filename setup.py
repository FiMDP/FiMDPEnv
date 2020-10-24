from setuptools import setup
import pathlib

from fimdpenv import __version__ as version

# The directory containing this file
HERE = pathlib.Path(__file__).parent
# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name='fimdpenv',
    version=version,
    description='Simulation environments for FiMDP',
    long_description=README,
    long_description_content_type="text/markdown",
    keywords='simulation-environment, grid-world, agent, cmdp, uuv-dynamics, street-network, formal-methods',
    url='https://github.com/FiMDP/FiMDPEnv',
    author="Pranay Thangeda",
    author_email="contact@prny.me",
    license="MIT",
    python_requires=">=3.6.0",
    packages=['fimdpenv'],
    install_requires=[
        'fimdp>=2.0.0',
        'numpy>=1.18.5',
        'matplotlib>=3.2.2',
        'scipy>=1.5.0',
        'ipython>=7.16.1',
        'networkx>=2.5',
        'folium>=0.11.0'
    ]
)




