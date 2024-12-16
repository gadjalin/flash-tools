import numpy as np
import re
from itertools import groupby
from collections.abc import Mapping, Sequence

class dat(object):
    """Stores a FLASH dat file

    This class reads and stores the data of a FLASH dat file.
    If the file contains multiple runs (usually starting with
    a new header, containing the columns' names), the data from each run
    are separated into different sub lists.
    """

    __columns = None
    __runs = None
    __loaded = None

    def __init__(self, filename = None):
        self.__columns = []
        self.__runs = []
        self.__loaded = False

        if filename is not None:
            self.read_file(filename)


    def check_loaded(self) -> bool:
        if not self.__loaded:
            raise RuntimeError('No dat file has been loaded yet!')


    def __getitem__(self, index: int | str | tuple[int, int] | tuple[int, str]):
        return self.get(index)


    def column_names(self):
        check_loaded()
        return self.__runs[0].dtype.names


    def get_run(self, run: int):
        """
        Returns the data for a specific run.
        
        Arguments
        ---
        run : int
            The index of the run.

        Returns
        ---
        An array containing the data of the specified run.
        Can be index using the appropriate column name.

        Raises
        ---
        RuntimeError
            If no file has been loaded in memory yet, either via the
            constructor or the read_file method.
        """

        self.check_loaded()
        return self.__runs[run]


    def get(self, index: int | str):
        """
        Returns the data of the specified column for all, or a specific
        run.

        Arguments
        ---
        index : int | str | tuple[int, int] | tuple[int, str]
            In the first case, either the index or the name of
            the column, and returns for all runs.
            Or a tuple containing first the index of the run, and
            second the index or name of the column.

        Returns
        ---
        The data from the specified column, for all runs or the
        specified one.

        Raises
        ---
        RuntimeError
            If no file has been loaded in memory yet, either via the
            constructor or the read_file method.
        IndexError
            If the index type is invalid.
        """

        self.check_loaded()
        # If index is int or str, retrieve data from all runs for corresponding columns
        if type(index) is int:
            return np.concatenate([np.atleast_1d(run[self.__runs[0].dtype.names[index]]) for run in self.__runs], axis=0)
        elif type(index) is str:
            return np.concatenate([np.atleast_1d(run[index]) for run in self.__runs], axis=0)
        # If tuple, retrieve column from specified run
        elif type(index) is tuple and len(index) == 2:
            run, column = index
            if type(column) is int:
                return np.atleast_1d(self.__runs[run][self.__runs[run].dtype.names[column]])
            elif type(column) is str:
                return np.atleast_1d(self.__runs[run][column])
        else:
            raise IndexError(f'Invalid index: {index}')


    def read_file(self, filename: str) -> None:
        """
        Reads the specified dat file and loads the data in memory.

        arguments
        --- 
        filename : str
            The path to the dat file.
        """

        with open(filename, 'r') as f:
            l = f.readline().strip()

        offset = 26
        width = 25
        self.__columns = []
        self.__columns.append(re.sub(r'^\d+\s*', '', l[1:width]).strip())

        while (offset + width < len(l)):
            self.__columns.append(re.sub(r'^\d+\s*', '', l[offset:offset+width].strip()).strip())
            offset += width+1

        self.__columns.append(re.sub(r'^\d+\s*', '', l[offset:].strip()).strip())

        skip = 0
        next_run = 0
        with open(filename, 'r') as f:
            for line in f:
                if (not line.strip().startswith('#')):
                    next_run += 1
                    continue

                if (next_run == 0):
                    skip += 1
                    continue

                # Parse file between two headers (one FLASH run)
                data = np.genfromtxt(
                    fname=filename,
                    names=self.__columns,
                    dtype=None,
                    encoding='ascii',
                    skip_header=skip,
                    max_rows=next_run
                )
                self.__runs.append(data)

                skip += next_run + 1
                next_run = 0

            # Parse last run, from header to EOF
            data = np.genfromtxt(
                fname=filename,
                names=self.__columns,
                dtype=None,
                encoding='ascii',
                skip_header=skip
            )
            self.__runs.append(data)

        self.__loaded = True


class model(object):
    """Opens a FLASH 1d progenitor profile

    It reads the names of the variables stored in the profile,
    and loads the data in each column.
    """

    __comment = None
    __var_names = None
    __data = None
    __loaded = None

    def __init__(self, filename = None):
        self.__comment = ""
        self.__var_names = []
        self.__data = []
        self.__loaded = False

        if filename is not None:
            self.read_file(filename)


    def check_loaded(self) -> bool:
        if not self.__loaded:
            raise RuntimeError('No model file has been loaded yet!')


    def var_names(self):
        """
        Returns the name of the columns.
        """

        check_loaded()
        return self.__data.dtype.names


    def __getitem__(self, index: int | str):
        return self.get(index)


    def get(self, index: int | str):
        """
        Returns the data in the specified column.

        Arguments
        ---
        index : int | str
            Index of the column, either by its name, or numerical index.

        Returns
        ---
        The data from the specified column.

        Raises
        ---
        RuntimeError
            If no file has been loaded in memory yet, either via the
            constructor or the read_file method.
        IndexError
            If the index type is invalid.
        """

        check_loaded()
        if type(index) is int:
            return self.__data[self.__data.dtype.names[index]]
        elif type(index) is str:
            return self.__data[index]
        else:
            raise IndexError(f'Invalid index: {index}')


    def read_file(self, filename: str) -> None:
        """
        Reads the specified profile file and loads the data in memory.

        arguments
        --- 
        filename : str
            The path to the 1d profile file.
        """

        skip = 0
        with open(filename, 'r') as f:
            line = f.readline()
            # Read comment on first line if any
            if (line.startswith('#')):
                skip += 1
                self.__comment = line
                line = f.readline()
            # Read "number of variables" line
            num_vars = int(line.split()[-1])
            skip += num_vars + 1

            # Read variables names
            self.__var_names = ['r']
            self.__var_names += [f.readline().split()[0] for i in range(num_vars)]

        # Load columns
        self.__data = np.genfromtxt(
            fname=filename,
            skip_header=skip,
            names=self.__var_names,
            dtype=None,
            encoding='ascii'
        )

        self.__loaded = True


def write_flash_profile(r: Sequence[float], data: Mapping[str, Sequence[float]], comment: str = "", output: str = "model.1d") -> None:
    """
    Create a new FLASH 1d profile with the specified data.

    Arguments
    ---
    r : list[float]
        An array containing the mid-cell radius of each zone.
    data : dict[str, list[float]]
        A dictionary mapping the variables' names to their values at
        radii given by r.
    comment : str
        A string containing a comment to put at the beginning of the
        file (without the leading # character).
    output : str
        The name of the file where to write the profile.
    """

    with open(output, 'w') as f:
        print("#", comment, file=f)
        print("number of variables =", len(data), file=f)

        for var in data:
            print(var, file=f)

        for i in range(len(r)):
            print(r[i], *[data[var][i] for var in data], file=f)


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
    'max_shock_radius': r'$r_\mathrm{sh}$', \str 
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

def calculate_shock(time, shock_rad):
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
    for k, g in groupby(max_shock_rads):
        offset += len(list(g))
        shock_times_smooth.append(times[offset - 1])
        shock_rad_smooth.append(k)

    #t_bounce = time[np.min(np.nonzero(max_shock_rad)) - 2]
    #shock_times = np.logspace(np.log10(t_bounce), np.log10(time[-1]), 1000)
    shock_times = np.linspace(time[0], time[-1], 1000)
    shock_rad = np.interp(shock_times, shock_times_smooth, shock_rad_smooth)
    shock_vel = np.gradient(shock_rad, shock_times)
    return shock_times, shock_rad, shock_vel

