import numpy as np
import re
from itertools import groupby

class dat(object):
    """
    Opens a FLASH dat file and separates in different groups if the file contains results from multiple runs.
    """
    def __init__(self, filename=None):
        self.columns = []
        self.runs = []
        self.loaded = False
        if filename is not None:
            self.read_file(filename)


    def check_loaded(self):
        if not self.loaded:
            raise RuntimeError('No dat file has been loaded yet!')


    def __getitem__(self, index: int | str | tuple[int, int] | tuple[int, str]):
        self.check_loaded()

        # If index is int or str, retrieve data from all runs for corresponding columns
        if type(index) is int:
            return np.concatenate([np.atleast_1d(run[self.runs[0].dtype.names[index]]) for run in self.runs], axis=0)
        elif type(index) is str:
            return np.concatenate([np.atleast_1d(run[index]) for run in self.runs], axis=0)
        # If tuple, retrieve column from specified run
        elif type(index) is tuple and len(index) == 2:
            run, column = index
            if type(column) is int:
                return np.atleast_1d(self.runs[run][self.runs[run].dtype.names[column]])
            elif type(column) is str:
                return np.atleast_1d(self.runs[run][column])


    def column_names(self):
        return self.columns


    def get_run(self, run: int):
        self.check_loaded()
        return self.runs[run]


    def get_column(self, column: int | str):
        self.check_loaded()
        if type(column) is int:
            return np.concatenate([np.atleast_1d(run[self.runs[0].dtype.names[index]]) for run in self.runs], axis=0)
        elif type(column) is str:
            return np.concatenate([np.atleast_1d(run[index]) for run in self.runs], axis=0)
        else:
            raise IndexError(f'Incorrect index: {index}')


    def read_file(self, filename):
        with open(filename, 'r') as f:
            l = f.readline().strip()

        offset = 26
        width = 25
        self.columns = []
        self.columns.append(re.sub(r'^\d+\s*', '', l[1:width]).strip())

        while (offset + width < len(l)):
            self.columns.append(re.sub(r'^\d+\s*', '', l[offset:offset+width].strip()).strip())
            offset += width+1

        self.columns.append(re.sub(r'^\d+\s*', '', l[offset:].strip()).strip())

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
                    names=self.columns,
                    dtype=None,
                    encoding='ascii',
                    skip_header=skip,
                    max_rows=next_run
                )
                self.runs.append(data)

                skip += next_run + 1
                next_run = 0

            # Parse last run, from header to EOF
            data = np.genfromtxt(
                fname=filename,
                names=self.columns,
                dtype=None,
                encoding='ascii',
                skip_header=skip
            )
            self.runs.append(data)

        self.loaded = True


class model(object):
    def __init__(self, filename=None):
        self.vars = []
        self.data = {}
        if filename is not None:
            self.read_file(filename)

    def __getitem__(self, var):
        return self.data[var]

    def read_file(self, filename):
        with open(filename, 'r') as f:
            skip = 0
            l = f.readline()
            if (l.startswith('#')):
                skip += 1
                l = f.readline()
            num_vars = int(l.split()[-1])
            skip += num_vars + 1

            self.vars = ['r']
            self.vars += [f.readline().split()[0] for i in range(num_vars)]
            
        self.data = np.genfromtxt(
            fname=filename,
            skip_header=skip,
            names=self.vars,
            dtype=None,
            encoding='ascii'
        )


def write_flash_profile(r, data, comment="", output="profile.flash"):
    with open(output, 'w') as output_file:
        print("#", comment, file=output_file)
        print("number of variables =", len(data), file=output_file)
        
        for var in data:
            print(var, file=output_file)
    
        for i in range(len(r)):
            print(r[i], *[data[var][i] for var in data], file=output_file)

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

def get_plot_label(var, normal=0, log=True):
    label = ''
    if var in _VAR_LABELS:
        label = _VAR_LABELS[var]
    else:
        label = var
    
    if normal != 0 and log:
        label += ' ['
        label += f'$\\times 10^{{{normal}}}$'
        if var in _VAR_UNITS:
            label += ' '
        else:
            label += ']'
    elif normal != 1 and not log:
        label += ' ['
        label += f'$\\times {normal}$'
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

def should_plot_log(var):
    if var in _VAR_LOGS:
        return _VAR_LOGS[var]
    else:
        return False

def get_bounce_time(logfile) -> float:
    with open(logfile, 'r') as file:
        for line in file:
            if 'Bounce!' in line:
                line = line.strip()
                return float(line.split()[1])
    return None

def calculate_shell_mass(r, dr, dens):
    return (4./3.) * np.pi * ((r + dr*0.5)**3 - (r - dr*0.5)**3) * dens

def calculate_shock(time, max_shock_rad):
    shock_times_smooth = [0]
    shock_rad_smooth = [0]
    offset = 0
    for k, g in groupby(max_shock_rad):
        offset += len(list(g))
        shock_times_smooth.append(time[offset - 1])
        shock_rad_smooth.append(k)

    #t_bounce = time[np.min(np.nonzero(max_shock_rad)) - 2]
    #shock_times = np.logspace(np.log10(t_bounce), np.log10(time[-1]), 1000)
    shock_times = np.linspace(time[0], time[-1], 1000)
    shock_rad = np.interp(shock_times, shock_times_smooth, shock_rad_smooth)
    shock_vel = np.gradient(shock_rad, shock_times)
    return shock_times, shock_rad, shock_vel
