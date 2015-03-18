#!/usr/bin/env python
'''
hybridLFPy
==========

Provides methods for estimating extracellular potentials of simplified spiking
neuron network models.


How to use the documentation
----------------------------

Documentation is available in two forms:
    1.  Docstrings provided with the code, e.g., as hybridLFPy? within IPython
    2.  Autogenerated sphinx-built output, compiled using
        ::
            
            sphinx-build -b html docs documentation
        
        in the root folder of the package sources


Available classes
-----------------
`CachedNetwork`
    Offline interface between network spike events and used by class Population

`CachedNoiseNetwork`
    Creation of Poisson spiketrains of a putative network model, interfaces
    class Population
    
`CachedFixedSpikesNetwork`
    Creation of identical spiketrains per population of a putative network
    model, interface to class Population

`GDF`
    Class using sqlite to efficiently store and enquire large amounts of spike
    output data, used by Cached*Network

`PopulationSuper`
    Parent class setting up a base population of multicompartment neurons

`Population`
    Daughter of PopulationSuper, using CachedNetwork spike events as synapse
    activation times with layer and cell type connection specificity

`PostProcess`
    Methods for processing output of multiple instances of class Population


Available utilities
-------------------
`csd`
    Ground truth current-source density estimation from multicompartment models

`helpers`
    Various methods used throughout simulations

`setup_file_dest`
    Setup destination folders of simulation output files

'''

from .cachednetworks import CachedNetwork, CachedNoiseNetwork, CachedFixedSpikesNetwork
from .gdf import GDF
from .population import PopulationSuper, Population
from .postproc import PostProcess
from . import csd
from . import helpers
from .helpers import setup_file_dest
