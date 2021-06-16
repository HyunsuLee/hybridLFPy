# Module `hybridLFPy`

Python module implementating a hybrid model scheme for predictions of
extracellular potentials (local field potentials, LFPs) of spiking
neuron network simulations.


## Project Status

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.45185.svg)](https://doi.org/10.5281/zenodo.45185)
[![Documentation Status](https://readthedocs.org/projects/hybridlfpy/badge/?version=latest)](https://hybridLFPy.readthedocs.io/en/latest/?badge=latest)
[![Upload Python Package](https://github.com/INM-6/hybridLFPy/workflows/Upload%20Python%20Package/badge.svg)](https://pypi.org/project/hybridLFPy)
[![Python pytest](https://github.com/INM-6/hybridLFPy/workflows/Python%20pytest/badge.svg)](https://github.com/INM-6/hybridLFPy/actions/workflows/python-pytest.yml)
[![License](http://img.shields.io/:license-GPLv3+-green.svg)](http://www.gnu.org/licenses/gpl-3.0.html)


##  Development

The module hybridLFPy was mainly developed in the Computational Neuroscience
Group (http://compneuro.umb.no), Department of Mathemathical Sciences and
Technology (http://www.nmbu.no/imt), at the Norwegian University of Life
Sciences (http://www.nmbu.no), Aas, Norway, in collaboration with Institute of
Neuroscience and Medicine (INM-6) and Institute for Advanced Simulation (IAS-6),
Juelich Research Centre and JARA, Juelich, Germany
(http://www.fz-juelich.de/inm/inm-6/EN/).


## Citation

Should you find `hybridLFPy` useful for your research, please cite the following paper:
```
Espen Hagen, David Dahmen, Maria L. Stavrinou, Henrik Lindén, Tom Tetzlaff,
Sacha J. van Albada, Sonja Grün, Markus Diesmann, Gaute T. Einevoll;
Hybrid Scheme for Modeling Local Field Potentials from Point-Neuron Networks,
Cerebral Cortex, Volume 26, Issue 12, 1 December 2016, Pages 4461–4496,
https://doi.org/10.1093/cercor/bhw237
```

Bibtex source:
```
@article{doi:10.1093/cercor/bhw237,
author = {Hagen, Espen and Dahmen, David and Stavrinou, Maria L. and Lindén, Henrik and Tetzlaff, Tom and van Albada, Sacha J. and Grün, Sonja and Diesmann, Markus and Einevoll, Gaute T.},
title = {Hybrid Scheme for Modeling Local Field Potentials from Point-Neuron Networks},
journal = {Cerebral Cortex},
volume = {26},
number = {12},
pages = {4461-4496},
year = {2016},
doi = {10.1093/cercor/bhw237},
URL = { + http://dx.doi.org/10.1093/cercor/bhw237},
eprint = {/oup/backfile/content_public/journal/cercor/26/12/10.1093_cercor_bhw237/2/bhw237.pdf}
}
```


## License

This software is released under the General Public License (see the [LICENSE](https://github.com/INM-6/hybridLFPy/blob/master/LICENSE) file).


## Warranty

This software comes without any form of warranty.


## Installation

First download all the `hybridLFPy` source files using `git`
(http://git-scm.com). Open a terminal window and type:
```
cd $HOME/where/to/put/hybridLFPy
git clone https://github.com/INM-6/hybridLFPy.git
```

To use `hybridLFPy` from any working folder without copying files, run:
```
(sudo) pip install -e . (--user)
```

Installing it is also possible, but not recommended as things might change with
future pulls from the repository:
```
(sudo) pip install . (--user)
```

### examples folder

Some example script(s) on how to use this module


### docs folder

Source files for autogenerated documentation using `Sphinx` (https://www.sphinx-doc.org).

To compile documentation source files in this directory using sphinx, use:
```
sphinx-build -b html docs documentation
```

### Dockerfile

The provided `Dockerfile` provides a Docker container recipe for `x86_64` hosts
with all dependencies required to run simulation files provided in `examples`.
To build and run the container locally, get Docker from https://www.docker.com
and issue the following (replace `<image-name>` with a name of your choosing):

    $ docker build -t <image-name> -< Dockerfile
    $ docker run -it -p 5000:5000 <image-name>:latest


The `--mount` option can be used to mount a folder on the host to a target folder as:

    $ docker run --mount type=bind,source="$(pwd)",target=/opt -it -p 5000:5000 <image-name>


## Online documentation

The sphinx-generated html documentation can be accessed at
https://hybridLFPy.readthedocs.io
