#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Hybrid LFP scheme example script, applying the methodology with the model of:

Potjans, T. and Diesmann, M. "The Cell-Type Specific Cortical Microcircuit:
Relating Structure and Activity in a Full-Scale Spiking Network Model".
Cereb. Cortex (2014) 24 (3): 785-806.
doi: 10.1093/cercor/bhs358

Synopsis of the main simulation procedure:
1. Loading of parameterset
    a. network parameters
    b. parameters for hybrid scheme
2. Set up file destinations for different simulation output
3. network simulation
    a. execute network simulation using NEST (www.nest-simulator.org)
    b. merge nest spike output from different MPI ranks
4. Create a object-representation that uses sqlite3 of all the spiking output
5. Iterate over post-synaptic populations:
    a. Create Population object with appropriate parameters for
       each specific population
    b. Run all computations for populations
    c. Postprocess simulation output of all cells in population
6. Postprocess all cell- and population-specific output data
7. Create a tarball for all non-redundant simulation output

The full simulation can be evoked by issuing a mpirun call, such as
    mpirun -np 64 python example_microcircuit.py
where the number 64 is the desired number of MPI threads & CPU cores

Given the size of the network and demands for the multi-compartment LFP-
predictions using the present scheme, running the model on a large scale
compute facility is strongly encouraged.

'''
from example_plotting import *
import matplotlib.pyplot as plt
from example_microcircuit_params import multicompartment_params, \
    point_neuron_network_params
import os
if 'DISPLAY' not in os.environ:
    import matplotlib
    matplotlib.use('Agg')
import numpy as np
from time import time
import neuron  # NEURON compiled with MPI must be imported before NEST and mpi4py
# to avoid NEURON being aware of MPI.
from hybridLFPy import PostProcess, Population, CachedNetwork
from hybridLFPy import setup_file_dest, helpers
from glob import glob
import tarfile
import lfpykit
from mpi4py import MPI
import nest


# set some seed values
SEED = 12345678
SIMULATIONSEED = 12345678
np.random.seed(SEED)


################# Initialization of MPI stuff ############################
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

# if True, execute full model. If False, do only the plotting. Simulation results
# must exist.
properrun = True


# check if mod file for synapse model specified in expisyn.mod is loaded
if not hasattr(neuron.h, 'ExpSynI'):
    if RANK == 0:
        os.system('nrnivmodl')
    COMM.Barrier()
    neuron.load_mechanisms('.')


##########################################################################
# PARAMETERS
##########################################################################


# Full set of parameters including network parameters
params = multicompartment_params()


##########################################################################
# Function declaration(s)
##########################################################################

def merge_gdf(model_params,
              raw_label='spikes_',
              file_type='gdf',
              fileprefix='spikes',
              skiprows=0):
    '''
    NEST produces one file per virtual process containing recorder output.
    This function gathers and combines them into one single file per
    network population.

    Parameters
    ----------
    model_params : object
        network parameters object
    raw_label : str
    file_type : str
    fileprefix : str
    skiprows : int

    Returns
    -------
    None

    '''
    def get_raw_gids(model_params):
        '''
        Reads text file containing gids of neuron populations as created within
        the NEST simulation. These gids are not continuous as in the simulation
        devices get created in between.

        Parameters
        ----------
        model_params : object
            network parameters object


        Returns
        -------
        gids : list
            list of neuron ids and value (spike time, voltage etc.)

        '''
        gidfile = open(os.path.join(model_params.raw_nest_output_path,
                                    model_params.GID_filename), 'r')
        gids = []
        for l in gidfile:
            a = l.split()
            gids.append([int(a[0]), int(a[1])])
        return gids

    # some preprocessing
    raw_gids = get_raw_gids(model_params)
    pop_sizes = [raw_gids[i][1] - raw_gids[i][0] + 1
                 for i in np.arange(model_params.Npops)]
    raw_first_gids = [raw_gids[i][0] for i in np.arange(model_params.Npops)]
    converted_first_gids = [int(1 + np.sum(pop_sizes[:i]))
                            for i in np.arange(model_params.Npops)]

    for pop_idx in np.arange(model_params.Npops):
        if pop_idx % SIZE == RANK:
            files = glob(os.path.join(model_params.raw_nest_output_path,
                                      raw_label + '{}*.{}'.format(pop_idx,
                                                                  file_type)))
            gdf = []  # init
            for f in files:
                new_gdf = helpers.read_gdf(f, skiprows)
                for line in new_gdf:
                    line[0] = line[0] - raw_first_gids[pop_idx] + \
                        converted_first_gids[pop_idx]
                    gdf.append(line)

            print(
                'writing: {}'.format(
                    os.path.join(
                        model_params.spike_output_path,
                        fileprefix +
                        '_{}.{}'.format(
                            model_params.X[pop_idx],
                            file_type))))
            helpers.write_gdf(
                gdf,
                os.path.join(
                    model_params.spike_output_path,
                    fileprefix +
                    '_{}.{}'.format(
                        model_params.X[pop_idx],
                        file_type)))

    COMM.Barrier()

    return


def dict_of_numpyarray_to_dict_of_list(d):
    '''
    Convert dictionary containing numpy arrays to dictionary containing lists

    Parameters
    ----------
    d : dict
        sli parameter name and value as dictionary key and value pairs

    Returns
    -------
    d : dict
        modified dictionary

    '''
    for key, value in d.items():
        if isinstance(value, dict):  # if value == dict
            # recurse
            d[key] = dict_of_numpyarray_to_dict_of_list(value)
        elif isinstance(value, np.ndarray):  # or isinstance(value,list) :
            d[key] = value.tolist()
    return d


def send_nest_params_to_sli(p):
    '''
    Read parameters and send them to SLI

    Parameters
    ----------
    p : dict
        sli parameter name and value as dictionary key and value pairs

    Returns
    -------
    None
    '''
    for name in list(p.keys()):
        value = p[name]
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, dict):
            value = dict_of_numpyarray_to_dict_of_list(value)
        if name == 'neuron_model':  # special case as neuron_model is a
            # NEST model and not a string
            try:
                nest.ll_api.sli_run('/' + name)
                nest.ll_api.sli_push(value)
                nest.ll_api.sli_run('eval')
                nest.ll_api.sli_run('def')
            except BaseException:
                print('Could not put variable %s on SLI stack' % (name))
                print(type(value))
        else:
            try:
                nest.ll_api.sli_run('/' + name)
                nest.ll_api.sli_push(value)
                nest.ll_api.sli_run('def')
            except BaseException:
                print('Could not put variable %s on SLI stack' % (name))
                print(type(value))
    return


def sli_run(parameters=object(),
            fname='microcircuit.sli',
            verbosity='M_INFO'):
    '''
    Takes parameter-class and name of main sli-script as input, initiating the
    simulation.

    Parameters
    ----------
    parameters : object
        parameter class instance
    fname : str
        path to sli codes to be executed
    verbosity : str,
        nest verbosity flag

    Returns
    -------
    None

    '''
    # Load parameters from params file, and pass them to nest
    # Python -> SLI
    send_nest_params_to_sli(vars(parameters))

    # set SLI verbosity
    nest.ll_api.sli_run("%s setverbosity" % verbosity)

    # Run NEST/SLI simulation
    nest.ll_api.sli_run('(%s) run' % fname)


def tar_raw_nest_output(raw_nest_output_path,
                        delete_files=True,
                        filepatterns=['voltages*.dat',
                                      'spikes*.dat',
                                      'weighted_input_spikes*.dat'
                                      '*.gdf']):
    '''
    Create tar file of content in `raw_nest_output_path` and optionally
    delete files matching given pattern.

    Parameters
    ----------
    raw_nest_output_path: path
        params.raw_nest_output_path
    delete_files: bool
        if True, delete .dat files
    filepatterns: list of str
        patterns of files being deleted
    '''
    if RANK == 0:
        # create tarfile
        fname = raw_nest_output_path + '.tar'
        with tarfile.open(fname, 'a') as t:
            t.add(raw_nest_output_path)

        # remove files from <raw_nest_output_path>
        for pattern in filepatterns:
            for f in glob(os.path.join(raw_nest_output_path, pattern)):
                try:
                    os.remove(f)
                except OSError:
                    print('Error while deleting {}'.format(f))

    # sync
    COMM.Barrier()

    return

###############################################################################
# MAIN simulation procedure
###############################################################################


# tic toc
tic = time()

if properrun:
    # set up the file destination
    setup_file_dest(params, clearDestination=True)

######## Perform network simulation ######################################

if properrun:
    # initiate nest simulation with only the point neuron network parameter
    # class
    networkParams = point_neuron_network_params()
    sli_run(parameters=networkParams,
            fname='microcircuit.sli',
            verbosity='M_INFO')

    # preprocess the gdf files containing spiking output, voltages, weighted and
    # spatial input spikes and currents:
    merge_gdf(networkParams,
              raw_label=networkParams.spike_recorder_label,
              file_type='dat',
              fileprefix=params.networkSimParams['label'],
              skiprows=3)

    # create tar file archive of <raw_nest_output_path> folder as .dat files are
    # no longer needed. Remove
    tar_raw_nest_output(params.raw_nest_output_path, delete_files=True)

# Create an object representation of the simulation output that uses sqlite3
networkSim = CachedNetwork(**params.networkSimParams)

toc = time() - tic
print('NEST simulation and gdf file processing done in  %.3f seconds' % toc)


# Set up LFPykit measurement probes for LFPs and CSDs
if properrun:
    probes = []
    probes.append(lfpykit.RecExtElectrode(cell=None, **params.electrodeParams))
    probes.append(
        lfpykit.LaminarCurrentSourceDensity(
            cell=None,
            **params.CSDParams))
    probes.append(lfpykit.CurrentDipoleMoment(cell=None))

####### Set up populations ###############################################

if properrun:
    # iterate over each cell type, run single-cell simulations and create
    # population object
    for i, y in enumerate(params.y):
        # create population:
        pop = Population(
            # parent class parameters
            cellParams=params.yCellParams[y],
            rand_rot_axis=params.rand_rot_axis[y],
            simulationParams=params.simulationParams,
            populationParams=params.populationParams[y],
            y=y,
            layerBoundaries=params.layerBoundaries,
            probes=probes,
            savelist=params.savelist,
            savefolder=params.savefolder,
            dt_output=params.dt_output,
            POPULATIONSEED=SIMULATIONSEED + i,
            # daughter class kwargs
            X=params.X,
            networkSim=networkSim,
            k_yXL=params.k_yXL[y],
            synParams=params.synParams[y],
            synDelayLoc=params.synDelayLoc[y],
            synDelayScale=params.synDelayScale[y],
            J_yX=params.J_yX[y],
            tau_yX=params.tau_yX[y],
            recordSingleContribFrac=params.recordSingleContribFrac,
        )

        # run population simulation and collect the data
        pop.run()
        pop.collect_data()

        # object no longer needed
        del pop

####### Postprocess the simulation output ################################


# reset seed, but output should be deterministic from now on
np.random.seed(SIMULATIONSEED)

if properrun:
    # do some postprocessing on the collected data, i.e., superposition
    # of population LFPs, CSDs etc
    postproc = PostProcess(y=params.y,
                           dt_output=params.dt_output,
                           probes=probes,
                           savefolder=params.savefolder,
                           mapping_Yy=params.mapping_Yy,
                           savelist=params.savelist
                           )

    # run through the procedure
    postproc.run()

    # create tar-archive with output
    postproc.create_tar_archive()

# tic toc
print('Execution time: %.3f seconds' % (time() - tic))

