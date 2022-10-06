#
#Programs to parallelize ebtelPlusPlus for batch runs
#
import sys
import os
from typing import Union, Optional
import numbers
import subprocess
import copy
from joblib import Parallel, delayed
import numpy as np

from astropy import units as u
import astropy.constants as const
import power_functions

# Set default top dir. Should point to the local top directory of ebtelPlusPlus
top_dir = '/mnt/c/Users/sschonfe/OneDrive - NASA/NASA/Research/ebtelPlusPlus/'


def heating_event(events, magnitude = 0, rise_start = 0, rise_end = 100,
                  decay_start = 100, decay_end = 200):
    """
    Program to construct a dictionary entry for a single heating event
    'events'        : dictionary containing ebtelPlusPlus heating events
    'magnitude'     : maximum heating rate of the event [erg cm^-3 s^-1]
    'rise_start'    : time the heating event starts [s]
    'rise_end'      : time the heating reaches its maximum [s]
    'decay_start'   : time the heating starts reducing [s]
    'rise_end'      : time the heating event ends [s]
    """
    if not isinstance(events,list):
        events = []

    events.append({'event':{'magnitude':magnitude,
                            'rise_start':rise_start,'rise_end':rise_end,
                            'decay_start':decay_start,'decay_end':decay_end}})
    return events


def heating_series(rate = 1e-3, time = 1e4, dt = 20, tau = 1000, bg = 1e-7):
    """
    Program to construct an evenly-spaced heating event series for ebtelPlusPlus
    'time'      : duration of the heating series [s]
    'rate'      : average heating rate [erg cm^-3 s^-1]
    'dt'        : individual heating event duration [s]
    'tau'       : constant time between heating events [s]
    'bg'        : background heating rate [erg cm^-3 s^-1]
    """
    # Determine start and end times of heating events
    nevents = int(np.ceil(time/tau))
    tstart = tau * np.arange(nevents)
    tstart[0] = -1  # Setting t<0 ensures loop is initiated with active heating
    tmid = tstart + dt/2
    tend = tstart + dt

    # determine amplitude of individual heating events
    if (rate - bg) > 0: # Check if the background greater than total heating
        if bg < 1e-2 * rate: # Adjust background to >= 1 % of total heating
            bg = 1e-2 * rate
        magnitude = (rate - bg) * (2 * tau / dt)
    else: # No impulsive heating if background greater than total heating rate
        return [], bg

    # Create heating event list
    events = []
    for ind in range(nevents):
        heating_event(events, magnitude = magnitude,
                       rise_start = tstart[ind], rise_end = tmid[ind],
                       decay_start = tmid[ind], decay_end = tend[ind])

    return events, bg


def heating_power_law(rate = 1e-3, time = 1e4, dt = 50,
                      tau = False, tau_min = False, tau_max = False,
                      alpha = -2.4, bg = 1e-7, rng = None):
    """
    Program to construct a distribution of heating events for ebtelPlusPlus
    'rate'      : average heating rate [erg cm^-3 s^-1]
    'time'      : duration of the heating series [s]
    'dt'        : individual heating event duration [s]
    'tau'       : characteristic delay between successive events [s]
    'tau_min'   : minimum delay time between successive events [s]
    'tau_max'   : maximum delay time between successive events [s]
    'alpha'     : slope of the power law distribution of heating event energies
    'bg'        : background heating rate [erg cm^-3 s^-1]
    'rng'       : numpy random number generator
    """
    events = []

    if (rate - bg) <= 0:  # No impulsive heating if background > total heating
        heating_event(events, magnitude = 0,
                      rise_start = 0, rise_end = 0,
                      decay_start = 0, decay_end = 0)
        return events, bg

    # Determine maximum delay time if not provided
    if tau_max == False:
        tau_max = 86400 # Default to maximum delay time of one day

    # Determine minimum delay time if neither 'tau_min' nor 'tau' provided
    if tau == False:
        if tau_min == False:
            tau_min = dt
    else:  # Determine heating delay limits based on characteristic 'tau'
        if tau_min == False:
            tau_min = max(0.1 * tau, dt)
        # Set heating event limits if characteristic delay 'tau' is defined
        tau_max, tau_min = power_functions.power_domain(alpha, tau,
                                                        tau_min, tau_max,
                                                        mean = False)

    # Sequentially generate a series of heating events
    events = []
    t_next = -1   # Setting t<0 ensures loop is initiated with active heating
    while t_next < time:
        # Generate random delay time until the next event
        delay = np.asscalar(power_functions.power_random(alpha,tau_min,tau_max,
                                                         rng=rng))

        # Adjust the last heating event to fit within the time series
        # Sets the next event to start just after the end of the time seires
        if t_next + delay + dt > time:
            delay = time - t_next

        # Calculate event heating based on delay time: rate = bg + (q / delay)
        q = delay * (rate - bg)
        magnitude = 2 * q / dt

        # Add event to event list
        heating_event(events, magnitude = magnitude,
                      rise_start = t_next, rise_end = t_next + dt / 2,
                      decay_start = t_next + dt / 2, decay_end = t_next + dt)

        # Increment time to next event
        t_next += delay

    return events, bg


def cooling_time(loop_length):
    """
    Returns the characteristic loop cooling time.
    Assumes: t_cool~loop_lengh and t_cool(loop half-length=35 Mm) = 2000 s

    'loop_length'   :   The coronal loop half-length in [cm]
    """
    tc = 2000/35e8  # Time constant such that 35 Mm half-loop cools in 2,000 s
    return loop_length * tc


def cooling_time_dynamic(half_length, average_heating_rate, duration,
                            max_event_factor):
    """
    Returns the characteristic loop cooling time.
    Estimates the loop cooling time derived primarily by those expressions
    given in the Appendix of Cargill (2014). From Barnes (2019):
    https://github.com/rice-solar-physics/synthetic-observables-paper-models/blob/80f68bceb7ecbcd238c196e3cc07d19e88617720/scripts/constrained_heating_model.py

    'half_length'           :   The coronal loop half-length in [cm]
    'average_heating_rate'  :   Coronal heating rate in [erg cm^-3 s^-1]
    'duration'              :   Triangular heating pulse duration [s]
    'max_event_factor'      :   Size of maximum event relative to the average
    """
    # set some constants
    alpha = -0.5
    chi = 6e-20  # *(u.erg*(u.cm**3)/u.s*u.K**(0.5))
    kappa_0 = 1e-6  # *(u.erg/u.cm/u.s*(u.K**(-7/2)))
    c1,c2,c3 = 2.0,0.9,0.6  # EBTEL constants
    gamma = 5./3.

    # approximate the characteristic heating event size
    average_heating_rate_max = (average_heating_rate * max_event_factor
                                * (cooling_time(half_length) / (2 * duration)))

    # estimate n0T0
    T0 = c2*(7.*half_length**2*average_heating_rate_max/2./kappa_0)**(2./7.)
    top_term = (average_heating_rate_max
                - (2.*kappa_0*(T0**(3.5))
                   /(7.*(c2**2.5)*c3*(half_length**2)*gamma)))
    bottom_term = c1*chi*(T0**alpha)*(1. - c2/c3/gamma)
    n0 = np.sqrt(top_term/bottom_term)
    n0T0 = n0*T0

    # Cargill cooling expression
    term1 = (2. - alpha)/(1. - alpha)
    term2 = (kappa_0**(4. - 2.*alpha))*(chi**7)
    term3 = ((half_length)**(8. - 4.*alpha))/(n0T0**(3+2.*alpha))

    return term1*3.*const.k_B.cgs.value*(1/term2*term3)**(1/(11. - 2.*alpha))


def radiative_loss_rate(T = False):
    """
    Program to calculate radiative loss rate using internal ebtel definition
    'T'     : a scalar, list, or np.ndarray of temperatures [K] to evaluate loss
    """
    # Check if temperature passed
    if not (isinstance(T, list) or isinstance(T, np.ndarray)):
        if T == False:
            print('ebtel_parallel.radiative_loss_rate ERROR: '
                    'no temperature supplied.')
            return False

    # Check if temperature has correct units
    if isinstance(T, u.quantity.Quantity):
        if T.unit == u.K:
            lgT = np.log10(T.value)
        else:
            print('ebtel_parallel.radiative_loss_rate ERROR: '
                    'temperature must have units of log(T [K]) or [K]')
            return False
    else:
        lgT = T
        print('ebtel_parallel.radiative_loss_rate WARNING: '
                'T is dimensionless. Assuming log_10(T)')

    # Compute loss function of temperature. Use boolean masks to make piecewise
    npfp = np.float_power # Redefine function for brevity
    loss = [                  (lgT <= 4.97)  * (1.09e-31 * npfp(10, 2 * lgT))
            + ((4.97 < lgT) & (lgT <= 5.67)) * (8.87e-17 * npfp(10, -lgT))
            + ((5.67 < lgT) & (lgT <= 6.18)) * (1.90e-22)
            + ((6.18 < lgT) & (lgT <= 6.55)) * (3.53e-13 * npfp(10, -1.5 * lgT))
            + ((6.55 < lgT) & (lgT <= 6.90)) * (3.46e-25 * npfp(10, lgT / 3))
            + ((6.90 < lgT) & (lgT <= 7.63)) * (5.49e-16 * npfp(10, -lgT))
            +  (7.63 < lgT)                  * (1.96e-27 * npfp(10, 0.5 * lgT))]

    return np.asarray(loss).ravel() * u.erg * (u.cm ** 3) / u.s


def calculate_ddm(filename=None, top_dir = top_dir):
    """
    Calculate the differential dispersion measure (DDM) of ebtelPlusPlus run
    Based on a combination of Jim Klimchuk's "ebtel2dd.pro" and the calculation
    of the dem in ebtelPlusPlus "dem.cpp".
    'filename'  : The configuration file from which ebtelPlusPlus was run.
                  The results, coronal DEM, and transition region DEM files
                  must all exist in the places where the configuration file
                  says they should be.
                  All necessary files and values are read in using this file.
    'top_dir'   : ebtelplusplus directory for importing rsp_toolkit
    returns     : (ddm_corona, ddm_tr) the coronal and transition region
                  differential dispersion measures (DDM(t,T) = DEM(t,T)/n(t))
                  with the same convention and shape as the ebtelplusplus
                  dem(t+1,T)
    """
    # Imports and system parameters
    sys.path.append(os.path.join(top_dir,'rsp_toolkit/python'))
    from xml_io import InputHandler

    # Define physical constants
    k_B = 1.380649*(10**(-16))      # Boltzmann's constant [erg/K]
    m_p = 1.672621923*(10**(-24))   # Proton mass [g]
    g_s = 2.74*(10**4)              # Surface gravity of the Sun [cm/s^2]

    # Read in configuration xml file
    ih = InputHandler(filename)
    xml = ih.lookup_vars()

    # Extract values from config_file
    length = xml['loop_length']

    # Calculate k_B and m_p corrections based on Helium abundance
    z_avg = ((1 + 2*xml['helium_to_hydrogen_ratio'])
            / (1 + xml['helium_to_hydrogen_ratio']))
    # k_B_correct = k_B / z_avg  # ebtellplusplus version
    k_B_correct = k_B * (0.5*(1+1/z_avg))  # ebtel2.pro version
    ion_mass_correction = ((1 + 4*xml['helium_to_hydrogen_ratio'])
                            / (2 + 3*xml['helium_to_hydrogen_ratio'])
                            * (1 + z_avg) / z_avg)
    m_p_correct = m_p * ion_mass_correction

    # Read output parameter file
    results = np.loadtxt(xml['output_filename'])
    n = copy.copy(results[:,3])     # plasma density
    T_e = copy.copy(results[:,1])   # electron temperature
    T_i = copy.copy(results[:,2])   # ion temperature
    p_e = copy.copy(results[:,4])   # electron pressure
    del results

    # Read output dem files
    dem_tr = np.loadtxt(xml['output_filename']+'.dem_tr')
    temp = copy.copy(dem_tr[0,:]) # The first row contains temperatures
    dem_corona = np.loadtxt(xml['output_filename'] + '.dem_corona')

    # Calculate coronal DDM
    ddm_corona = np.zeros(dem_corona.shape)
    ddm_corona[0,:] = dem_corona[0,:]
    ddm_corona[1:,:] = dem_corona[1:,:]/n[:,None]
    del dem_corona

    # Calculate transition region gravitational scale height with temperature
    scale_height = (k_B * (T_e + T_i/z_avg)
                    / m_p_correct
                    / (xml['surface_gravity']*g_s))

    # Compute transition region DDM
    p_e_array = np.broadcast_to(p_e[:,None], dem_tr[1:,:].shape)
    temp_array = np.broadcast_to(temp[None,:], dem_tr[1::].shape)
    scale_height_array = np.broadcast_to(scale_height[:,None], dem_tr[1::].shape)
    n_tr = ((p_e_array / k_B_correct / temp_array)
            * np.exp(2 * length * np.sin(np.pi/5) / scale_height_array / np.pi))
    ddm_tr = np.zeros(dem_tr.shape)
    ddm_tr[0,:] = dem_tr[0,:]
    ddm_tr[1:,:] = dem_tr[1:,:] / n_tr
    del dem_tr

    return ddm_corona, ddm_tr


def run_ebtel(filename = False, top_dir = top_dir):
    """
    Program to implement a single ebtelPlusPlus run
    'filename'  : name of .cfg.xml file with default ebtelPlusPlus parameters
    'top_dir'   : directory of ebtelplusplus directory for importing rsp_toolkit
    """
    if filename == False:
        print('ebtel_parallel.run_ebtel ERROR: no .xml parameter file passed. '
              + 'Ebtel not run.')
        return False
    if os.path.isfile(filename) == False:
        print('ebtel_parallel.run_ebtel ERROR: '
                + ' file does not exist. Ebtel not run.')
        return False

    # Define ebtel++ c++ program location
    python_exec = os.path.join(top_dir,'bin/ebtel++.run')

    # Call ebtelPlusPlus c++ program
    return subprocess.call([python_exec, '-c', filename])


def run_ebtel_repeat(xml, repeats = 1, run = 0, stable = 0.5,
                     filename = 'ebtel.parallel.cfg.xml',
                     heating = 'constant', top_dir = top_dir,
                     cleanup = True):
    """
    Program to implement a single ebtelPlusPlus run for a repeated solution
    'xml'       : xml list of input parameters
    'run'       : index of repeated runs
    'stable'    : skip fraction or first index of results to calculate
                  DEM integral
    'filename'  : name of .cfg.xml file with default ebtelPlusPlus parameters
    'heating'   : flag for type of heating ('constant' or 'power')
    'top_dir'   : ebtelplusplus directory for importing rsp_toolkit
    'cleanup'   : binary toggle to delete temporary run files when finished
    """
    # Imports and system parameters
    sys.path.append(os.path.join(top_dir,'rsp_toolkit/python'))
    from xml_io import InputHandler, OutputHandler

    # Read in default configuration file
    ih = InputHandler(filename)
    xml = ih.lookup_vars()

    # Define values from inputs
    im_name = filename.replace('.cfg.xml', '.png')
    length = xml['loop_length']
    heating_rate = xml['energy'] / length
    try:
        background = xml['heating']['background']
    except:
        background = 1e-7

    # Set heating event time-delay parameters
    try:
        tau_scale = xml['frequency_scale']
        assert (isinstance(tau_scale, numbers.Number) and (tau_scale != 0))
        try:    # Retrieve relative size of largest heating event
            max_factor = xml['max_event_factor']
            assert (isinstance(max_factor,numbers.Number) and (max_factor >= 1))
        except: # Defualt bahvior, largest event 3 times larger than average
            max_factor = 3
        tau = cooling_time_dynamic(length, heating_rate, xml['duration'],
                                    max_factor)
        tau *= tau_scale
        tau_max  = tau * max_factor  # Maximum heating event delay
        tau_min = 1e-7 * tau  # Will cause tau_min to be set automatically
    except:
        tau = xml['delay']
        tau_max = xml['delay_maximum']
        tau_min = xml['delay_minimum']

    # Set total simulation time
    try: # Try to calculate based on xml <cooling_times>
        total_time = xml['cooling_times'] * max(tau, xml['duration']) # tau
    except:
        try: # If not set, default to xml <total_time>
            total_time = xml['total_time'] # Default to xml <total_time>
        except: # Fallback, heat loops for 100 cooling times
            total_time = 100 * tau

    # Calculate heating events
    if heating == 'constant':
        events, bkg = heating_series(rate = heating_rate,
                                     time = total_time,
                                     dt = xml['duration'],
                                     tau = tau,
                                     bg = background)
    elif heating == 'power':
        events, bkg = heating_power_law(rate = heating_rate,
                                        time = total_time,
                                        dt = xml['duration'],
                                        tau = tau,
                                        tau_min = tau_min,
                                        tau_max = tau_max,
                                        alpha = xml['alpha'],
                                        bg = background)
    else:
        print('ebtel_parallel.run_ebtel_repeat ERROR: '
                + 'heating must be "constant" or "power"')
        return False

    # Modify xml file parameters
    xml['total_time'] = total_time
    xml['tau_max'] = tau_max  # Different tau_max, this sets maximum timestep
    xml['delay'] = tau
    if heating =='power':
        xml['delay_minimum'] = tau_min
        xml['delay_maximum'] = tau_max
    xml['heating']['background'] = bkg
    xml['heating']['events'] = events
    xml['output_filename'] = (xml['output_filename']
                              + '_' + str(xml['loop_length']/1e8)
                              + '_' + str(xml['energy']/1e7)
                              + '_' + str(xml['delay'])
                              + '_' + str(run))
    xml_name = xml['output_filename'] + '.xml'

    # Write out the configuration file
    oh = OutputHandler(xml_name, xml)
    oh.print_to_xml()

    # Call ebtelPlusPlus c++ program
    run_ebtel(xml_name)

    # Read in results time series
    results = np.loadtxt(xml['output_filename'])

    #Handle the 'stable' input
    stbl = abs(stable)
    if stbl < 1:
        stbl = int(stbl * xml['total_time'])
    stbl = np.argmax(results[:,0] >= stbl) # Select 1st index after stable time

    # Start plotting output series and DEM for run 0
    if run == 0:
        import matplotlib.pyplot as plt
        from matplotlib import gridspec
        from matplotlib.ticker import MaxNLocator

        # Setup figure
        fig = plt.figure(figsize=(7,4))
        gs = gridspec.GridSpec(3,2, width_ratios=[3,2])
        ax1 = fig.add_subplot(gs[0,0])
        ax2 = fig.add_subplot(gs[1,0], sharex=ax1)
        ax3 = fig.add_subplot(gs[2,0], sharex=ax1)
        ax4 = fig.add_subplot(gs[:,1])

        # Plot evolution of plasma parameters
        ax1.plot(results[:,0],results[:,-1])
        ax1.axvline(x=results[stbl,0], color='C7')
        ax2.plot(results[:,0], results[:,2]/1e+6, label=r'$T_i$', color = 'C1')
        ax2.plot(results[:,0], results[:,1]/1e+6, label=r'$T_e$', color = 'C0')
        ax2.axvline(x=results[stbl,0], color='C7')
        ax3.plot(results[:,0], results[:,3]/1e+9)
        ax3.axvline(x=results[stbl,0], color='C7')

        # Plot axes options
        ax1.set_ylabel(r'$Q$ [erg cm$^{-3}$ s$^{-1}$]')
        ax1.tick_params(axis='x', labelbottom=False)
        ax1.yaxis.set_major_locator(MaxNLocator(prune='lower', nbins=5))
        ax2.set_ylabel(r'$T$ [MK]')
        ax2.tick_params(axis='x', labelbottom=False)
        ax2.yaxis.set_major_locator(MaxNLocator(prune='lower', nbins=5))
        ax3.set_ylabel(r'$n$ [$10^9$ cm$^{-3}$]')
        ax3.set_xlabel(r'$t$ [s]')
        ax3.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax3.set_xlim([results[0,0], results[-1,0]])

    #Record relevant information from results array and clear the memory
    tmax = np.amax(results[stbl:,1]) #Maximum electron temperature during run
    weights = np.gradient(results[stbl:,0]) # DEM weighting by time step length
    n_ave = np.average(results[stbl:,3], axis=0, weights=weights)
    t_ave = np.average(results[stbl:,1], axis=0, weights=weights)
    tn_ave = np.average(results[stbl:,1], axis=0,
                        weights=weights*results[stbl:,3])
    tn2_ave = np.average(results[stbl:,1], axis=0,
                         weights=weights*(results[stbl:,3]**2))
    del results

    #Read in and time average transition region DEM time series
    dem_tr = np.loadtxt(xml['output_filename']+'.dem_tr')
    demt = np.copy(dem_tr[0,:]) # The first row contains temperatures
    tr_ave = np.average(dem_tr[stbl+1:,:], axis=0, weights=weights)
    del dem_tr
    tr_ave = tr_ave / 2 # Average DEM over the two footpoints

    #Read in and time average coronal DEM time series
    dem_corona = np.loadtxt(xml['output_filename'] + '.dem_corona')
    corona_ave = np.average(dem_corona[stbl+1:,:], axis=0, weights=weights)
    del dem_corona
    corona_ave = corona_ave / (2 * length) # Average DEM over the two loop legs

    # Finish plotting output series and DEM
    if run == 0:
        # Plot output DEMs
        ax4.step(demt, tr_ave, where = 'mid', color='C0',
                 label=r'$\mathrm{DEM}_{\mathrm{TR}}$')
        ax4.step(demt, corona_ave * length, where = 'mid', color='C1',
                 label=r'$\mathrm{DEM}_{\mathrm{C}}$')
        ax4.step(demt, tr_ave + corona_ave * length, where = 'mid', color='C7',
                 label='DEM')

        # Plot axes options
        ax4.set_xlabel(r'$T$ [K]')
        ax4.set_ylabel(r'$\mathrm{DEM}$ [cm$^{-5}$ K$^{-1}$]')
        ax4.set_xlim([demt[0], demt[-1]])
        ymax = np.amax([tr_ave + corona_ave * length])
        ax4.set_ylim([1e-3*ymax, ymax])
        ax4.set_xscale('log')
        ax4.set_yscale('log')
        ax4.legend(loc='best', labelspacing=0)

        # Finish plot
        plt.subplots_adjust(hspace=0, wspace=0.325,
                            bottom=0.125, top=0.98, left=0.115, right=0.99)
        plt.savefig(im_name, dpi=600)
    else: # Clean up after yourself (delete output files, return relevant data)
        if cleanup:
            os.remove(xml['output_filename'])
            os.remove(xml['output_filename']+'.dem_tr')
            os.remove(xml['output_filename'] + '.dem_corona')
            os.remove(xml['output_filename'] + '.xml')

    return dict(run = run, dem_tr = tr_ave, dem_cor = corona_ave, tmax = tmax,
                n_ave=n_ave, t_ave=t_ave, tn_ave=tn_ave, tn2_ave=tn2_ave)


def epp_repeat(file='ebtel_parallel_test', procs = -1, runs = 1,
                   heating = 'constant', stable = 0.5, top_dir = top_dir,
                   cleanup=True):
    """
    Program to perform repeated batch ebtelPlusPlus runs in parallel on CPU
    'file'      : file base (without extensions) to read and save for this run
    'procs'     : number of processor threads to use in the computation [-1=all]
    'runs'      : number of times to repeat the run
    'heating'   : flag for type of heating ('constant' or 'power')
    'stable'    : point in run to consider it stable. Step number or fraction.
    'top_dir'   : ebtelplusplus directory for importing rsp_toolkit
    'cleanup'   : binary toggle to delete run files when finished
    """
    sys.path.append(os.path.join(top_dir,'rsp_toolkit/python'))
    from xml_io import InputHandler, OutputHandler

    # Alert user which file is being worked on
    print('ebtel_parallel.epp_cpu_repeat '
            + f'computing {runs} iterations from {file}')

    #Read in ebtelPlusPlus .xml file
    try:
        file = str(file)
        file_cfg = file + '.cfg.xml'
        ih = InputHandler(file_cfg)
    except FileNotFoundError:
        print(f'ebtel_parallel.epp_cpu ERROR: {file_cfg} file not found.')
        return False
    params = ih.lookup_vars()

    # Make arrays to hold outputs of grid calculations
    logtdem = np.linspace(params['dem']['temperature']['log_min'],
                        params['dem']['temperature']['log_max'],
                        params['dem']['temperature']['bins'])
    dem_tr = np.zeros((runs, params['dem']['temperature']['bins']))
    dem_cor = copy.copy(dem_tr)
    n_arr = np.zeros(runs)
    t_arr = copy.copy(n_arr)
    tn_arr = copy.copy(n_arr)
    tn2_arr = copy.copy(n_arr)

    # Perform cpu parallelized ebtelPlusPlus runs
    ps = Parallel(n_jobs=procs, verbose=5)(delayed(run_ebtel_repeat)
                        (params, repeats = runs, run = run, filename = file_cfg,
                         heating = heating, stable = stable, top_dir = top_dir)
                        for run in range(0,runs))

    # Parse results from parallel runs into output dictionary
    for rn in ps:
        dem_tr[rn['run'],:] = rn['dem_tr']
        dem_cor[rn['run'],:] = rn['dem_cor']
        n_arr[rn['run']] = rn['n_ave']
        t_arr[rn['run']] = rn['t_ave']
        tn_arr[rn['run']] = rn['tn_ave']
        tn2_arr[rn['run']] = rn['tn2_ave']

    del ps

    n_ave = np.average(n_arr)
    n_std = np.std(n_arr)
    t_ave = np.average(t_arr)
    t_std = np.std(t_arr)
    tn_ave = np.average(tn_arr)
    tn_std = np.std(tn_arr)
    tn2_ave = np.average(tn2_arr)
    tn2_std = np.std(tn2_arr)

    # Save results to npz file
    np.savez(file, params = params,
             logtdem = logtdem, dem_tr = dem_tr, dem_cor = dem_cor,
             n_ave = n_ave, n_std = n_std, t_ave = t_ave, t_std = t_std,
             tn_ave=tn_ave, tn_std=tn_std, tn2_ave=tn2_ave, tn2_std=tn2_std)

    return True


def run_ebtel_grid(xml, ll, hh, stable = 0.5,
                   filename = 'ebtel.parallel.cfg.xml',
                   heating = 'constant', heat_params = [], top_dir = top_dir,
                   cleanup=True):
    """
    Program to implement a single ebtelPlusPlus run for a grid of solutions
    'xml'       : xml list of input parameters
    'll'        : index of length loop
                    (length = 10 ** (6 + 0.1 * ll) [cm])
    'hh'        : index of heating loop
                    (heat = 10 ** (-3 + (dd * hh) - (dd * ll)) [erg cm^-3 s^-1])
    'stable'    : starting fraction or first index of results to start
                  DEM integral
    'filename'  : name of .cfg.xml file with default ebtelPlusPlus parameters
    'heating'   : flag for type of heating ('constant' or 'power')
    'heat_params'   : parameters to pass to the heating
    'top_dir'   : ebtelplusplus directory for importing rsp_toolkit
    'cleanup'   : binary toggle to delete run files when finished
    """
    # Imports and system parameters
    sys.path.append(os.path.join(top_dir,'rsp_toolkit/python'))
    from xml_io import InputHandler, OutputHandler

    # Read in default configuration file
    ih = InputHandler(filename)
    xml = ih.lookup_vars()
    im_name = filename.replace('.cfg.xml', '.png')

    # Calculate loop length and time averaged heating rate
    dd = 0.1 # log spacing of solutions
    length = 10 ** (6 + dd * ll)  # log by dd in ll from ~10 km
    heat = 10 ** (-3 + (dd * hh) - (dd * ll))  # For 1e3-8e8 erg cm^-2 s^-1

    # Calculate background heating rate
    long_loops = 1e10    # Loop length in [cm]
    bkg_const =  6.3e12  # Energy rate for loops with T_cor~3*10^5 K [erg s^-1 ]
    bkg_min = 1e-7 # Minimum background heat flux [erg cm^-2 s^-1]
    ideal_bkg = bkg_const / (length**2) # for T_cor~3*10^5 K
    ideal_bkg = max(ideal_bkg, bkg_min) # Minimum heating for stable loops
    heat_bkg = min(heat, ideal_bkg)  # Background heating up to "heat"

    # Set heating event time-delay parameters
    try:
        tau_scale = xml['frequency_scale']
        assert (isinstance(tau_scale, numbers.Number) and (tau_scale != 0))
        try:    # Retrieve relative size of largest heating event
            max_factor = xml['max_event_factor']
            assert (isinstance(max_factor,numbers.Number) and (max_factor >= 1))
        except: # Defualt bahvior, largest event 3 times larger than average
            max_factor = 3
        tau = cooling_time_dynamic(length, heat, xml['duration'], max_factor)
        tau *= tau_scale
        tau_max  = tau * max_factor  # Maximum heating event delay
        tau_min = 1e-7 * tau  # Will cause tau_min to be set automatically
    except:
        tau = xml['delay']
        tau_max = xml['delay_maximum']
        tau_min = xml['delay_minimum']

    # Catch fully background heating scenarios to shorten simulation times
    if heat_bkg >= heat:
        total_time = max(xml['duration'], tau)  # Minimum, length of one event
    else:
        try:
            total_time = xml['cooling_times'] * max(tau, xml['duration']) # tau
        except:  # Default to heating loops for 100 cooling times
            total_time = 100 * tau

    # Calculate heating events
    if heating == 'constant':
        events, bkg = heating_series(rate = heat,
                                     time = total_time,
                                     dt = xml['duration'],
                                     tau = tau,
                                     bg = heat_bkg)
    elif heating == 'power':
        try:
            if xml['random_seed'] == 0:
                seed = None
            else:
                seed = xml['random_seed']
        except:
            seed = None
        finally:
            rng = np.random.default_rng(seed)

        events, bkg = heating_power_law(rate = heat,
                                        time = total_time,
                                        dt = xml['duration'],
                                        tau = tau,
                                        tau_min = tau_min,
                                        tau_max = tau_max,
                                        alpha = xml['alpha'],
                                        bg = heat_bkg,
                                        rng = rng)
    else:
        print('ebtel_parallel.run_ebtel_grid ERROR: '
                + 'heating must be "constant" or "power"')
        return False

    # Modify xml file parameters
    xml['total_time'] = total_time
    xml['loop_length'] = length
    xml['tau_max'] = tau_max  # Different tau_max, this sets maximum timestep
    xml['energy'] = length * heat
    xml['delay'] = tau
    xml['delay_minimum'] = tau_min
    xml['delay_maximum'] = tau_max
    xml['heating']['background'] = heat_bkg
    xml['heating']['events'] = events
    xml['output_filename'] = (xml['output_filename']
                              + '_' + str(hh) + '_' + str(ll))
    xml_name = xml['output_filename'] + '.xml'

    # Write out the configuration file
    oh = OutputHandler(xml_name, xml)
    oh.print_to_xml()

    # Call ebtelPlusPlus c++ program
    msg = run_ebtel(xml_name)
    # Attempt to make grid runs robust to failure. This does not work well
    # while msg != 0: # If run fails, retry with looser condition.
    #     xml['adaptive_solver_error'] *= 10
    #     xml['adaptive_solver_safety'] += 0.9*(1-xml['adaptive_solver_safety'])
    #     # Write out the configuration file
    #     oh = OutputHandler(xml_name, xml)
    #     oh.print_to_xml()
    #     print(f"retrying hh={hh} and ll={ll} with "
    #             + f"threshold={xml['adaptive_solver_error']}")
    #     msg = run_ebtel(xml_name)

    # Read in results time series
    results = np.loadtxt(xml['output_filename'])

    #Handle the 'stable' input
    stbl = abs(stable)
    if stbl < 1:
        stbl = int(stbl * xml['total_time'])
    stbl = np.argmax(results[:,0] > stbl) # Select first index after stable time

    # Start plotting output series and DEM for l=10 Mm, h=10^-3 erg cm^-3 s^-1
    plot_length = 1e9
    plot_heat = 1e-3
    if (length == plot_length) and (heat == plot_heat):
        import matplotlib.pyplot as plt
        from matplotlib import gridspec
        from matplotlib.ticker import MaxNLocator

        # Setup figure
        fig = plt.figure(figsize=(7,4))
        gs = gridspec.GridSpec(3,2)
        ax1 = fig.add_subplot(gs[0,0])
        ax2 = fig.add_subplot(gs[1,0], sharex=ax1)
        ax3 = fig.add_subplot(gs[2,0], sharex=ax1)
        ax4 = fig.add_subplot(gs[:,1])

        # Plot evolution of plasma parameters
        ax1.plot(results[:,0],results[:,-1])
        ax1.axvline(x=results[stbl,0], color='C7')
        ax2.plot(results[:,0], results[:,2]/1e+6, label=r'$T_i$', color = 'C1')
        ax2.plot(results[:,0], results[:,1]/1e+6, label=r'$T_e$', color = 'C0')
        ax2.axvline(x=results[stbl,0], color='C7')
        ax3.plot(results[:,0], results[:,3]/1e+9)
        ax3.axvline(x=results[stbl,0], color='C7')

        # Plot axes options
        ax1.set_ylabel(r'$Q$ [erg cm$^{-3}$ s$^{-1}$]')
        ax1.tick_params(axis='x', labelbottom=False)
        ax1.yaxis.set_major_locator(MaxNLocator(prune='lower', nbins=5))
        ax2.set_ylabel(r'$T$ [MK]')
        ax2.tick_params(axis='x', labelbottom=False)
        ax2.yaxis.set_major_locator(MaxNLocator(prune='lower', nbins=5))
        ax3.set_ylabel(r'$n$ [$10^9$ cm$^{-3}$]')
        ax3.set_xlabel(r'$t$ [s]')
        ax3.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax3.set_xlim([results[0,0], results[-1,0]])
        # ax2.legend(loc='best', labelspacing=0)

    #Record relevant information from results array and clear the memory
    tmax = np.amax(results[stbl:,1]) #Maximum electron temperature during run
    weights = np.gradient(results[stbl:,0]) # DEM weighting by time step length
    del results

    #Read in and time average transition region DEM time series
    dem_tr = np.loadtxt(xml['output_filename']+'.dem_tr')
    demt = np.copy(dem_tr[0,:]) # The first row contains temperatures
    tr_ave = np.average(dem_tr[stbl+1:,:], axis=0, weights=weights)
    del dem_tr
    tr_ave = tr_ave / 2 # Average DEM over the two footpoints

    #Read in and time average coronal DEM time series
    dem_corona = np.loadtxt(xml['output_filename'] + '.dem_corona')
    corona_ave = np.average(dem_corona[stbl+1:,:], axis=0, weights=weights)
    del dem_corona
    corona_ave = corona_ave / (2 * length) # Average DEM over the two loop legs

    # Compute and time average transition region and coronal DDM time series
    ddm_cor, ddm_tr = calculate_ddm(xml['output_filename']+'.xml')
    ddm_tr_ave = np.average(ddm_tr[stbl+1:,:], axis=0, weights=weights)
    ddm_tr_ave = ddm_tr_ave / 2 # Average DDM over the two footpoints
    del ddm_tr
    ddm_cor_ave = np.average(ddm_cor[stbl+1:,:], axis=0, weights=weights)
    del ddm_cor
    ddm_cor_ave = ddm_cor_ave / (2*length) # Average DDM over the two loop legs

    # Finish plotting output series and DEM for l=10 Mm, h=10^-3 erg cm^-3 s^-1
    if (length == plot_length) and (heat == plot_heat):

        # Plot output DEMs
        ax4.step(demt, tr_ave, where='mid', color='C0',
                 label=r'$\mathrm{DEM}_{\mathrm{TR}}$')
        ax4.step(demt, corona_ave * length, where='mid', color='C1',
                 label=r'$\mathrm{DEM}_{\mathrm{C}}$')
        ax4.step(demt, tr_ave + corona_ave * length, where = 'mid',
                 color='black',
                 label=r'$\mathrm{DEM}_{\mathrm{total}}$')
        # Plot output DDMs
        ax5 = ax4.twinx()
        ax5.step(demt, ddm_tr_ave, where='mid', color='C2',
                 label=r'$\mathrm{DDM}_{\mathrm{TR}}$')
        ax5.step(demt, ddm_cor_ave * length, where='mid', color='C3',
                 label=r'$\mathrm{DDM}_{\mathrm{C}}$')
        ax5.step(demt, ddm_tr_ave + ddm_cor_ave * length, where = 'mid',
                 color='C7',
                 label=r'$\mathrm{DDM}_{\mathrm{total}}$')

        # Plot axes options
        ax4.set_xlabel(r'$T$ [K]')
        ax4.set_ylabel(r'$\mathrm{DEM}$ [cm$^{-5}$ K$^{-1}$]', color='black')
        ax5.set_ylabel(r'$\mathrm{DDM}$ [cm$^{-2}$ K$^{-1}$]', color='C7')
        ax4.set_xlim([demt[0], demt[-1]])
        dem_max = np.amax([tr_ave + corona_ave * length])
        ax4.set_ylim([1e-3*dem_max, dem_max])
        ddm_max = np.amax([ddm_tr_ave + ddm_cor_ave * length])
        ax5.set_ylim([1e-3*ddm_max, ddm_max])
        ax4.set_xscale('log')
        ax4.set_yscale('log')
        ax5.set_yscale('log')
        ax4.legend(loc='best', labelspacing=0)

        # Finish plot
        plt.subplots_adjust(hspace=0, wspace=0.325,
                            bottom=0.125, top=0.98, left=0.115, right=0.9)
        plt.savefig(im_name, dpi=600)

    # Clean up after yourself (delete output files, relevant data is returned)
    if True:
        if (ll % 10 == 0) and (hh % 10 == 0):
            print('Saving time profiles of hh={} and ll={}'.format(hh,ll))
        else:
            if cleanup:
                os.remove(xml['output_filename'])
                os.remove(xml['output_filename']+'.dem_tr')
                os.remove(xml['output_filename'] + '.dem_corona')
                os.remove(xml['output_filename'] + '.xml')

    return dict(ll=ll, hh=hh, length=length, heat=heat,
                dem_tr=tr_ave, dem_cor=corona_ave,
                ddm_tr = ddm_tr_ave, ddm_cor = ddm_cor_ave,
                tmax = tmax)


def epp_grid(file='ebtel_parallel_test', procs = -1, heat = [50,1e4,9e-9],
                 heating = 'constant', stable = 0.5,
                 top_dir = top_dir, cleanup=True):
    """
    Program to perform batch ebtelPlusPlus grid runs in parallel on CPU
    'file'      : file base (without extensions) to read and save for this run
    'procs'     : number of processor threads to use in the computation [-1=all]
    'heat'      : parameters of simulated heating events [dt, tau].
                  Can also include tau_min, tau_max, and alpha for heating
                  distribution and 'bg' for consistent or distribution heating
    'heating'   : flag for type of heating ('constant' or 'power')
    'stable'    : point in run to consider it stable. Step number or fraction.
    'top_dir'   : ebtelplusplus directory for importing rsp_toolkit
    'cleanup'   : binary toggle to delete run files when finished
    """
    sys.path.append(top_dir + 'rsp_toolkit/python')
    from xml_io import InputHandler, OutputHandler

    #Read in ebtelPlusPlus .xml file
    try:
        file = str(file)
        file_cfg = file + '.cfg.xml'
        ih = InputHandler(file_cfg)
    except FileNotFoundError:
        print(f'ebtel_parallel.epp_grid ERROR: {file_cfg} file not found.')
        return False
    params = ih.lookup_vars()

    # Make grid 10 km to ~6e5 km, and <H>= 1e3 to ~8e8 erg cm^-2 s^-1
    # Step by 0.1 in log space
    # Ebtel_Diffuse_GG.pro covers 10^6-~3*10^10 [cm], ~10^-8--1 [erg cm^-3 s^-1]
    lmin = 0
    lmax = 48
    # length_range = range(lmin, lmax+1)
    length_range = range(lmax, lmin-1, -1)  # Same range, just start from maxima
    hmin = 0
    hmax = 59
    # heat_range = range(hmin, hmax+1)
    heat_range = range(hmax, hmin-1, -1)  # Same range, just start from maxima

    # Make arrays to hold outputs of grid calculations
    dem_t = np.linspace(params['dem']['temperature']['log_min'],
                        params['dem']['temperature']['log_max'],
                        params['dem']['temperature']['bins'])
    dem_arr = np.zeros((len(length_range), len(heat_range),
                        params['dem']['temperature']['bins']))
    p_space = np.zeros((len(length_range), len(heat_range)))

    # Make dictionary to hold all outputs from grid calculations
    output = {'ll':length_range,'hh':heat_range,
              'dem_tr':dem_arr.copy(), 'dem_cor':dem_arr.copy(),
              'ddm_tr':dem_arr.copy(), 'ddm_cor':dem_arr.copy(),
              'logtdem':dem_t, 'lrun':p_space.copy(),
              'qrun':p_space.copy(), 'trun':p_space.copy()}
    del dem_t
    del dem_arr
    del p_space

    # Perform cpu parallelized ebtelPlusPlus runs
    ps = Parallel(n_jobs=procs, verbose=5)(delayed(run_ebtel_grid)
                        (params, ll = i, hh = j, filename = file_cfg,
                         heat_params = heat, heating = heating,
                         stable = stable, top_dir = top_dir, cleanup=cleanup)
                        for j in heat_range for i in length_range)

    # Parse results from parallel runs into output dictionary
    for rn in ps:
        if rn['ll'] in output['ll'] and rn['hh'] in output['hh']:
            lind = rn['ll'] - lmin
            hind = rn['hh'] - hmin
            output['dem_tr'][lind,hind,:] = rn['dem_tr']
            output['dem_cor'][lind,hind,:] = rn['dem_cor']
            output['ddm_tr'][lind,hind,:] = rn['ddm_tr']
            output['ddm_cor'][lind,hind,:] = rn['ddm_cor']
            output['lrun'][lind,hind] = rn['length']
            output['qrun'][lind,hind] = rn['heat']
            output['trun'][lind,hind] = rn['tmax']
        else:
            print('Failed to save output for hh={}, ll={}'.format(
                    rn['hh'],rn['ll']))
    del ps

    # Save results to npz file
    np.savez(file, hh=output['hh'], ll=output['ll'],
             dem_tr=output['dem_tr'], dem_cor=output['dem_cor'],
             logtdem=output['logtdem'], lrun=output['lrun'],
             qrun=output['qrun'], trun=output['trun'])

    # Write results to tab separated column text file to be read into IDL
    with open(file + '_params.txt', 'w') as f:
        f.write('hh\tll\tlrun\tqrun\ttrun')
        for j,hh in enumerate(output['hh']):
            for i,ll in enumerate(output['ll']):
                f.write('\n{}\t{}\t{}\t{}\t{}'.format(hh, ll,
                                                        output['lrun'][i,j],
                                                        output['qrun'][i,j],
                                                        output['trun'][i,j]))

    with open(file + '_dems.txt', 'w') as d:
        d.write('hh\tll\tlogt\tdem_tr\tdem_cor\tddm_tr\tddm_cor')
        for j,hh in enumerate(output['hh']):
            for i,ll in enumerate(output['ll']):
                for t,logt in enumerate(output['logtdem']):
                    d.write('\n{}\t{}\t{:1.2f}\t{}\t{}\t{}\t{}'.format(
                            hh, ll, logt,
                            output['dem_tr'][i,j,t], output['dem_cor'][i,j,t],
                            output['ddm_tr'][i,j,t], output['ddm_cor'][i,j,t]))

    return True


def epp_gpu_grid():
    """
    Program to perform batch ebtelPlusPlus runs in parallel on GPU
    """
    # CURRENTLY EMPTY.
    # THIS WAS A LONG TERM PLAN WHEN I STARTED. MAYBE YOU WANT TO WRITE IT?
    return False

# Behaviour if called as a script
if __name__ == "__main__":
    ebtel_parallel.epp_grid(file='runs/demo_grid', heating='power', stable=0.1)
