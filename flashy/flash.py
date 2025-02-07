import numpy as np
import re
from itertools import groupby
from collections.abc import Mapping, Sequence


_VAR_LABELS = { \
    'time':             r'$t$', \
    'r':                r'$r$', \
    'mass':             r'$M$', \
    'dens':             r'$\rho$', \
    'temp':             r'$T$', \
    'eint':             r'$e_\mathrm{int}$', \
    'ener':             r'$e_\mathrm{tot}$', \
    'velx':             r'$v$', \
    'vrad':             r'$v_\mathrm{rad}$', \
    'eexp':             r'$E_\mathrm{exp}$', \
    'max_shock_radius': r'$r_\mathrm{sh}$', \
    'shock_vel':        r'$v_\mathrm{sh}$', \
    'explosion_energy': r'$E_\mathrm{exp}$', \
    'point_mass':       r'$m_\mathrm{point}$', \
    'pres':             r'Pressure', \
    'ye':               r'$Y_e$', \
    'sumy':             r'SumY', \
    'abar':             r'$\mathcal{\bar{A}}$', \
    'zbar':             r'$\mathcal{\bar{Z}}$', \
}

_VAR_UNITS = {
    'time':             r'$\mathrm{s}$', \
    'r':                r'$\mathrm{cm}$', \
    'mass':             r'$M_\odot$', \
    'dens':             r'$\mathrm{g\,cm^{-3}}$', \
    'temp':             r'$\mathrm{K}$', \
    'eint':             r'$\mathrm{erg\,g^{-1}}$', \
    'ener':             r'$\mathrm{erg\,g^{-1}}$', \
    'velx':             r'$\mathrm{cm\,s^{-1}}$', \
    'vrad':             r'$\mathrm{cm\,s^{-1}}$', \
    'eexp':             r'$\mathrm{erg}$', \
    'max_shock_radius': r'$\mathrm{cm}$', \
    'shock_vel':        r'$\mathrm{cm\,s^{-1}}$', \
    'explosion_energy': r'$\mathrm{erg}$', \
    'point_mass':       r'$\mathrm{g}$', \
    'pres':             r'$\mathrm{g\,cm^{-1}\,s^{-2}}$' \
}

_VAR_LOGS = {
    'time':             False, \
    'r':                True, \
    'dens':             True, \
    'temp':             True, \
    'eint':             True, \
    'ener':             True, \
    'velx':             False, \
    'vely':             False, \
    'velz':             False, \
    'vrad':             False, \
    'entr':             False, \
    'ye'  :             False, \
    'sumy':             False, \
    'pres':             True, \
    'max_shock_radius': True, \
}

def get_plot_label(var: str, normalise: int = 0, log: bool = True) -> str:
    """
    Returns an appropriate LaTeX label for use in a matplotlib plot.

    Arguments
    ---
    var : str
        Common variable name from a FLASH profile or plot file.
    normalise : int
        When are the data are normalised to a different scale.
        Use in combination with log. If log is True and normalise is
        different than 0, a 10^{normalise} is prepended to the units in
        the label.
        If log is False and normalise is different than 1, the value of
        the normalise variable is prepended to the units in the label.
    log : bool
        Use in combination with normalise.
        If log is True, the data are normalised on a log scale, and on
        a linear scale otherwise.

    TODO
    ---
    Automatically figure out simpler units e.g. print "GK"
    instead of "10^9 K".
    """

    label = ''
    if var in _VAR_LABELS:
        label = _VAR_LABELS[var]
    else:
        label = var

    if normalise != 0 and log:
        label += ' ['
        label += f'$\\times 10^{{{normalise}}}$'
        if var in _VAR_UNITS:
            label += ' '
        else:
            label += ']'
    elif normalise != 1 and not log:
        label += ' ['
        label += f'$\\times {normalise}$'
        if var in _VAR_UNITS:
            label += ' '
        else:
            label += ']'
    elif var in _VAR_UNITS:
        label += ' ['

    if var in _VAR_UNITS:
        label += _VAR_UNITS[var]
        label += ']'

    return label

def should_plot_log(var: str) -> bool:
    """
    Determines if a variable should preferably be plotted on
    a log scale.

    Arguments
    ---
    var : str
        Common variable name from a FLASH profile or plot file.

    Returns
    ---
    True if var should be on a log scale, False if linear or
    variable is unknown.
    """

    if var in _VAR_LOGS:
        return _VAR_LOGS[var]
    else:
        return False

def get_bounce_time(logfile: str) -> float:
    """
    Finds the exact bounce time from the log file.

    Arguments
    ---
    logfile : str
        Path to the log file of the simulation.

    Returns
    ---
        The bounce time in seconds, or None if not found.
    """

    with open(logfile, 'r') as f:
        for line in f:
            if 'Bounce!' in line:
                line = line.strip()
                return float(line.split()[1])
    return None

def calculate_shell_mass(r, dr, dens):
    """
    Calculates the mass of the shells.

    Arguments
    ---
    r : list[float]
        The mid-cell radial coordinates of the shells.
    dr : list[float]
        The width of the shells.
    dens : list[float]
        The average density in the shells.
    """

    return (4./3.) * np.pi * ((r + dr*0.5)**3 - (r - dr*0.5)**3) * dens

def calculate_shock(time, shock_radius):
    """
    Calculates shock velocity.

    Arguments
    ---
    time : list[float]
        The list of times at which the shock radius is evaluated.
    shock_rad : list[float]
        The shock radius at different times.
        The min|max|mean_shock_radius column from the dat file.

    Returns
    ---
    A tuple of lists of floats, containing the processed shock times,
    radii and velocity.
    """

    shock_times_smooth = [0]
    shock_rad_smooth = [0]
    offset = 0
    # The dat file usually contains duplicates of the same time.
    # Removes the duplicates for a smoother result.
    for k, g in groupby(shock_radius):
        offset += len(list(g))
        shock_times_smooth.append(time[offset - 1])
        shock_rad_smooth.append(k)

    #t_bounce = time[np.min(np.nonzero(max_shock_rad)) - 2]
    #shock_times = np.logspace(np.log10(t_bounce), np.log10(time[-1]), 1000)
    shock_times = np.linspace(time[0], time[-1], 1000)
    shock_rad = np.interp(shock_times, shock_times_smooth, shock_rad_smooth)
    shock_vel = np.gradient(shock_rad, shock_times)
    return shock_times, shock_rad, shock_vel

