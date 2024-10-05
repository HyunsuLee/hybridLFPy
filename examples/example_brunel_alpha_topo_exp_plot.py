#!/usr/bin/env python
'''
Hybrid LFP scheme example script, applying the methodology with a model
implementation similar to:

Nicolas Brunel. "Dynamics of Sparsely Connected Networks of Excitatory and
Inhibitory Spiking Neurons". J Comput Neurosci, May 2000, Volume 8,
Issue 3, pp 183-208

But the network is implemented with spatial connectivity, i.e., the neurons
are assigned positions and distance-dependent connectivity in terms of
cell-cell connectivity and transmission delays.

Synopsis of the main simulation procedure:
1. Loading of parameterset
    a. network parameters
    b. parameters for hybrid scheme
2. Set up file destinations for different simulation output
3. network simulation
    a. execute network simulation using NEST (www.nest-initiative.org)
    b. merge network output (spikes, currents, voltages)
4. Create a object-representation that uses sqlite3 of all the spiking output
5. Iterate over post-synaptic populations:
    a. Create Population object with appropriate parameters for
       each specific population
    b. Run all computations for populations
    c. Postprocess simulation output of all cells in population
6. Postprocess all cell- and population-specific output data
7. Create a tarball for all non-redundant simulation output

The full simulation can be evoked by issuing a mpirun call, such as
mpirun -np 4 python example_brunel_alpha_topo_exp.py

Not recommended, but running it serially should also work, e.g., calling
python example_brunel_alpha_topo_exp.py

Given the size of the network and demands for the multi-compartment LFP-
predictions using the present scheme, running the model on nothing but a large-
scale compute facility is strongly discouraged.
'''

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import brunel_alpha_nest_topo_exp as BN
from parameters import ParameterSet
from hybridLFPy import CachedTopoNetwork



# set up file destinations differentiating between certain output
PS = ParameterSet(dict(
    # Main folder of simulation output
    savefolder='simulation_output_example_brunel_topo_exp',

    # make a local copy of main files used in simulations
    sim_scripts_path=os.path.join('simulation_output_example_brunel_topo_exp',
                                  'sim_scripts'),

    # destination of single-cell output during simulation
    cells_subfolder='cells',
    cells_path=os.path.join(
        'simulation_output_example_brunel_topo_exp',
        'cells'),

    # destination of cell- and population-specific signals, i.e.,
    # compund LFPs, CSDs etc.
    populations_subfolder='populations',
    populations_path=os.path.join('simulation_output_example_brunel_topo_exp',
                                  'populations'),

    # location of spike output from the network model
    spike_output_path=BN.spike_output_path,

    # destination of figure file output generated during model execution
    figures_subfolder='figures',
    figures_path=os.path.join('simulation_output_example_brunel_topo_exp',
                              'figures'),

))


# population (and cell type) specific parameters
PS.update(dict(
    # no cell type specificity within each E-I population
    # hence X == x and Y == X
    X=["EX", "IN"],

    # population-specific LFPy.Cell parameters
    cellParams=dict(
        # excitory cells
        EX=dict(
            morphology='morphologies/ex.swc',
            v_init=BN.neuron_params['E_L'],
            cm=1.0,
            Ra=150,
            passive=True,
            passive_parameters=dict(g_pas=1. / (BN.neuron_params['tau_m']
                                                * 1E3),  # assume cm=1
                                    e_pas=BN.neuron_params['E_L']),
            nsegs_method='lambda_f',
            lambda_f=100,
            dt=BN.dt,
            tstart=0,
            tstop=BN.simtime,
            verbose=False,
        ),
        # inhibitory cells
        IN=dict(
            morphology='morphologies/in.swc',
            v_init=BN.neuron_params['E_L'],
            cm=1.0,
            Ra=150,
            passive=True,
            passive_parameters=dict(g_pas=1. / (BN.neuron_params['tau_m']
                                                * 1E3),
                                    e_pas=BN.neuron_params['E_L']),
            nsegs_method='lambda_f',
            lambda_f=100,
            dt=BN.dt,
            tstart=0,
            tstop=BN.simtime,
            verbose=False,
        )),

    # assuming excitatory cells are pyramidal
    rand_rot_axis=dict(
        EX=['z'],
        IN=['x', 'y', 'z'],
    ),

    # kwargs passed to LFPy.Cell.simulate()
    simulationParams=dict(),

    # set up parameters corresponding to model populations, the x-y coordinates
    # will use position-data from network simulation, but sliced if more than
    # one cell type y is assigned to represent a main population Y
    populationParams=dict(
        EX=dict(
            number=BN.NE,
            position_index_in_Y=['EX', 0],
            z_min=-450,
            z_max=-350,
        ),
        IN=dict(
            number=BN.NI,
            position_index_in_Y=['IN', 0],
            z_min=-450,
            z_max=-350,
        ),
    ),

    # set the boundaries between the "upper" and "lower" layer
    layerBoundaries=[[0., -300],
                     [-300, -500]],

    # set the geometry of the virtual recording device
    electrodeParams=dict(
        # contact locations:
        x=np.meshgrid(np.linspace(-1800, 1800, 10),
                      np.linspace(-1800, 1800, 10))[0].flatten(),
        y=np.meshgrid(np.linspace(-1800, 1800, 10),
                      np.linspace(-1800, 1800, 10))[1].flatten(),
        z=[-400. for x in range(100)],
        # extracellular conductivity:
        sigma=0.3,
        # contact surface normals, radius, n-point averaging
        N=[[0, 0, 1]] * 100,
        r=5,
        n=20,
        seedvalue=None,
        # dendrite line sources, soma as sphere source (Linden2014)
        method='root_as_point',
    ),

    # runtime, cell-specific attributes and output that will be stored
    savelist=[
    ],

    # time resolution of saved signals
    dt_output=BN.dt * 2
))


# for each population, define layer- and population-specific connectivity
# parameters
PS.update(dict(
    # number of connections from each presynaptic population onto each
    # layer per postsynaptic population, preserving overall indegree
    k_yXL=dict(
        EX=[[int(0.5 * BN.CE), 0],
            [int(0.5 * BN.CE), BN.CI]],
        IN=[[0, 0],
            [BN.CE, BN.CI]],
    ),

    # set up table of synapse PSCs from each possible connection
    J_yX=dict(
        EX=[BN.J_ex * 1E-3, BN.J_in * 1E-3],
        IN=[BN.J_ex * 1E-3, BN.J_in * 1E-3],
    ),

    # set up synapse parameters as derived from the network
    synParams=dict(
        EX=dict(
            section=['apic', 'dend'],
            tau=BN.tauSyn,
            syntype='AlphaISyn'
        ),
        IN=dict(
            section=['dend'],
            tau=BN.tauSyn,
            syntype='AlphaISyn'
        ),
    ),

    # set up delays, here using fixed delays of network
    synDelayLoc=dict(
        EX=[None, None],
        IN=[None, None],
    ),
    # no distribution of delays
    synDelayScale=dict(
        EX=[None, None],
        IN=[None, None],
    ),


    # For topology-like connectivity. Only exponential connectivity and
    # circular masks are supported with fixed indegree given by k_yXL,
    # using information on extent and edge wrap (periodic boundaries).
    # At present, synapse delays are not distance-dependent. For speed,
    # multapses are always allowed.
    # Information is here duplicated for each postsynaptic population as in the
    # network, but it could potentially be set per postsynaptic population
    topology_connections=dict(
        EX={
            X: dict(
                extent=BN.layerdict_EX['extent'],
                edge_wrap=BN.layerdict_EX['edge_wrap'],
                allow_autapses=BN.conn_dict_EX['allow_autapses'],
                kernel=BN.conn_kernel_EX,
                mask=BN.conn_dict_EX['mask'],
                delays=BN.conn_delay_EX,
            ) for X in ['EX', 'IN']},
        IN={
            X: dict(
                extent=BN.layerdict_IN['extent'],
                edge_wrap=BN.layerdict_IN['edge_wrap'],
                allow_autapses=BN.conn_dict_IN['allow_autapses'],
                kernel=BN.conn_kernel_IN,
                mask=BN.conn_dict_IN['mask'],
                delays=BN.conn_delay_IN,
            ) for X in ['EX', 'IN']},
    )

))


# putative mappting between population type and cell type specificity,
# but here all presynaptic senders are also postsynaptic targets
PS.update(dict(
    mapping_Yy=list(zip(PS.X, PS.X))
))

# Create an object representation containing the spiking activity of the
# network simulation output that uses sqlite3. Again, kwargs are derived from
# the brunel network instance.
networkSim = CachedTopoNetwork(
    simtime=BN.simtime,
    dt=BN.dt,
    spike_output_path=BN.spike_output_path,
    label=BN.label,
    ext='dat',
    GIDs={'EX': [1, BN.NE], 'IN': [BN.NE + 1, BN.NI]},
    label_positions=BN.label_positions,
    cmap='bwr_r',
    skiprows=3,
)

##########################################################################
# Create set of plots from simulation output
##########################################################################

def network_activity_animation(PS, networkSim,
                               T=(0, 200), kernel=np.exp(-np.arange(10) / 2),
                               save_anim=True):
    '''network activity animation'''
    fig, ax = plt.subplots(1, figsize=(9, 10))
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.15, top=0.975)

    ax.set_aspect('equal')

    ax.set_xlim(
        (-BN.layerdict_EX['extent'][0] / 2,
         BN.layerdict_EX['extent'][0] / 2))
    ax.set_ylim(
        (-BN.layerdict_EX['extent'][1] / 2,
         BN.layerdict_EX['extent'][1] / 2))
    ax.set_xlabel('x (um)', labelpad=0)
    ax.set_ylabel('y (um)', labelpad=0)
    ax.set_title('t=%.3i ms' % 100)

    dt = PS.dt_output
    tbins = np.arange(T[0], T[1] + dt, dt)

    spikes = {}
    scat = {}
    for j, X in enumerate(PS.X):
        db = networkSim.dbs[X]
        gid = networkSim.nodes[X]
        gid_t = np.asarray(db.select_neurons_interval(gid, T), dtype=object)

        spikes[X] = np.zeros(gid_t.shape[0],
                             dtype=[('pos', float, 2),
                                    ('size', float, tbins.size - 1)])
        # set position arrays
        spikes[X]['pos'] = networkSim.positions[X]
        # set size arrays
        for i, t in enumerate(gid_t):
            spikes[X]['size'][i, :] = np.convolve(
                np.histogram(t, bins=tbins)[0] * 200, kernel, 'same')

        # scatter plot of positions, will not be shown in animation
        scat[X] = ax.scatter(
            spikes[X]['pos'][:, 0], spikes[X]['pos'][:, 1],
            s=np.random.rand(spikes[X]['size'].shape[0]) * 100,
            facecolors=networkSim.colors[j], edgecolors='none', label=X)

    # set legend
    ax.legend(loc=(0.65, -0.2), ncol=3, fontsize=10, frameon=False)

    def update(frame_number):
        '''update function for animation'''
        ind = frame_number % (tbins.size - 1)
        for j, X in enumerate(PS.X):
            scat[X].set_sizes(spikes[X]['size'][:, ind])
        ax.set_title('t=%.3i ms' % tbins[ind])

    ani = FuncAnimation(fig, update, frames=tbins.size, interval=1)
    if save_anim:
        ani.save(os.path.join(PS.savefolder, 'NetworkTopo.mp4'),
                 fps=15, writer='ffmpeg',
                 extra_args=['-b:v', '5000k', '-r', '25', '-vcodec', 'mpeg4'],)

    # plt.show()


def lfp_activity_animation(PS, networkSim,
                           T=(0, 200), kernel=np.exp(-np.arange(10) / 2),
                           save_anim=True):
    '''animation of network activity and LFP data'''
    fig, ax = plt.subplots(1, figsize=(9, 10))
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.15, top=0.975)
    cbax = fig.add_axes([0.4, 0.1, 0.2, 0.02])

    ax.set_aspect('equal')

    ax.set_xlim(
        (-BN.layerdict_EX['extent'][0] / 2,
         BN.layerdict_EX['extent'][0] / 2))
    ax.set_ylim(
        (-BN.layerdict_EX['extent'][1] / 2,
         BN.layerdict_EX['extent'][1] / 2))
    ax.set_xlabel('x (um)', labelpad=0)
    ax.set_ylabel('y (um)', labelpad=0)
    ax.set_title('t=%.3i ms' % 100)

    dt = PS.dt_output
    tbins = np.arange(T[0], T[1] + dt, dt)

    # electrode geometry
    ax.scatter(PS.electrodeParams['x'], PS.electrodeParams['y'],
               s=20, color='k')

    # LFP data
    fname = os.path.join(PS.savefolder, 'PeriodicLFP_sum.h5')
    f = h5py.File(fname)
    data = f['data'][()]
    # subtract mean
    dataT = data.T - data.mean(axis=1)
    data = dataT.T
    # reshape
    data = data.reshape(
        (int(np.sqrt(PS.electrodeParams['x'].size)), -1, data.shape[-1]))

    # draw image plot on axes
    im = ax.pcolormesh(np.r_[0:4001:400] -
                       2000, np.r_[0:4001:400] -
                       2000, data[:, :, 0], vmin=-
                       data.std() *
                       4, vmax=data.std() *
                       4, zorder=-
                       1, cmap='jet_r')

    cbar = plt.colorbar(im, cax=cbax, orientation='horizontal')
    cbar.set_label('LFP (mV)', labelpad=0)
    tclbls = cbar.ax.get_xticklabels()
    plt.setp(tclbls, rotation=90, fontsize=10)

    def update(frame_number):
        '''update function for animation'''
        ind = frame_number % (tbins.size - 1)
        im.set_array(data[:, :, ind].flatten())
        ax.set_title('t=%.3i ms' % tbins[ind])

    ani = FuncAnimation(fig, update, frames=tbins.size, interval=1)
    if save_anim:
        ani.save(
            os.path.join(
                PS.savefolder,
                'LFPTopo.mp4'),
            fps=15,
            writer='ffmpeg',
            extra_args=[
                '-b:v',
                '5000k',
                '-r',
                '25',
                '-vcodec',
                'mpeg4'],
        )
    # plt.show()


def network_lfp_activity_animation(PS, networkSim, T=(
        0, 200), kernel=np.exp(-np.arange(10) / 2), save_anim=True):
    '''animation of network activity and LFP data'''
    fig, ax = plt.subplots(1, figsize=(9, 10))
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.15, top=0.975)
    cbax = fig.add_axes([0.4, 0.1, 0.2, 0.02])

    ax.set_aspect('equal')

    ax.set_xlim(
        (-BN.layerdict_EX['extent'][0] / 2,
         BN.layerdict_EX['extent'][0] / 2))
    ax.set_ylim(
        (-BN.layerdict_EX['extent'][1] / 2,
         BN.layerdict_EX['extent'][1] / 2))
    ax.set_xlabel('x (um)', labelpad=0)
    ax.set_ylabel('y (um)', labelpad=0)
    ax.set_title('t=%.3i ms' % 100)

    dt = PS.dt_output
    tbins = np.arange(T[0], T[1] + dt, dt)

    spikes = {}
    scat = {}
    for j, X in enumerate(PS.X):
        db = networkSim.dbs[X]
        gid = networkSim.nodes[X]
        gid_t = np.asarray(db.select_neurons_interval(gid, T), dtype=object)

        spikes[X] = np.zeros(gid_t.shape[0],
                             dtype=[('pos', float, 2),
                                    # dtype=[('pos', float,
                                    # networkSim.positions[X].shape),
                                    ('size', float, tbins.size - 1)])
        # set position arrays
        spikes[X]['pos'] = networkSim.positions[X]
        # set size arrays
        for i, t in enumerate(gid_t):
            spikes[X]['size'][i, :] = np.convolve(
                np.histogram(t, bins=tbins)[0] * 200, kernel, 'same')

        # scatter plot of positions, will not be shown in animation
        scat[X] = ax.scatter(
            spikes[X]['pos'][:, 0], spikes[X]['pos'][:, 1],
            s=np.random.rand(spikes[X]['size'].shape[0]) * 100,
            facecolors=networkSim.colors[j], edgecolors='none', label=X)

    # set legend
    ax.legend(loc=(0.65, -0.2), ncol=3, fontsize=10, frameon=False)

    # electrode geometry
    ax.scatter(PS.electrodeParams['x'], PS.electrodeParams['y'],
               s=20, color='k')

    # LFP data
    fname = os.path.join(PS.savefolder, 'PeriodicLFP_sum.h5')
    f = h5py.File(fname)
    data = f['data'][()]
    # subtract mean
    dataT = data.T - data.mean(axis=1)
    data = dataT.T
    # reshape
    data = data.reshape(
        (int(np.sqrt(PS.electrodeParams['x'].size)), -1, data.shape[-1]))

    # draw image plot on axes
    im = ax.pcolormesh(np.r_[0:4001:400] -
                       2000, np.r_[0:4001:400] -
                       2000, data[:, :, 0], vmin=-
                       data.std() *
                       4, vmax=data.std() *
                       4, zorder=-
                       1, cmap='jet_r')

    cbar = plt.colorbar(im, cax=cbax, orientation='horizontal')
    cbar.set_label('LFP (mV)', labelpad=0)
    tclbls = cbar.ax.get_xticklabels()
    plt.setp(tclbls, rotation=90, fontsize=10)

    def update(frame_number):
        '''update function for animation'''
        ind = frame_number % (tbins.size - 1)
        for j, X in enumerate(PS.X):
            scat[X].set_sizes(spikes[X]['size'][:, ind])
        im.set_array(data[:, :, ind].flatten())
        ax.set_title('t=%.3i ms' % tbins[ind])

    ani = FuncAnimation(fig, update, frames=tbins.size, interval=1)
    if save_anim:
        ani.save(
            os.path.join(
                PS.savefolder,
                'hybridLFPyTopo.mp4'),
            fps=15,
            writer='ffmpeg',
            extra_args=[
                '-b:v',
                '5000k',
                '-r',
                '25',
                '-vcodec',
                'mpeg4'],
        )
    # plt.show()
#

network_activity_animation(PS, networkSim, save_anim=True)
lfp_activity_animation(PS, networkSim, save_anim=True)
network_lfp_activity_animation(PS, networkSim, save_anim=True)

