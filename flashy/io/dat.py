import numpy as np
import re


class dat(object):
    """
    Open a FLASH dat file

    This class reads and stores the simulation output
    from a FLASH dat file.
    If the file contains multiple runs (usually starting with
    a new header, containing the columns' names),
    the data from each run are split into different lists.
    """

    __columns: list
    __runs: list
    __loaded: bool

    def __init__(self, filename = None):
        self.__columns = []
        self.__runs = []
        self.__loaded = False

        if filename is not None:
            self.readfile(filename)

    @classmethod
    def fromfile(cls, filename):
        obj = cls(filename)
        return obj

    def __checkloaded(self) -> bool:
        if not self.__loaded:
            raise RuntimeError('No dat file has been loaded yet!')

    def __getitem__(self, index):
        if isinstance(index, tuple):
            return self.get(*index)
        else:
            return self.get(index)

    def columns(self):
        """
        Return the name of the columns.
        """
        return self.__columns

    def runs(self):
        """
        How many runs this dat file contains.
        """
        return len(self.__runs)

    def alldata(self, run: int | list | tuple | np.ndarray = None):
        """
        Return all columns for all, or a specific run(s).
        
        Parameters
        ----------
        run : int
            The index of a run, or a list of runs.

        Returns
        -------
        An array containing the data of the specified run(s).
        Can be indexed using the appropriate column name.

        Raises
        ------
        RuntimeError
            If no file has been loaded in memory yet, either via the
            constructor or the readfile method.
        IndexError
            If the index type is invalid.
        """

        self.__checkloaded()
        if run is None:
            return np.concatenate(self.__runs)
        elif isinstance(run, int):
            return self.__runs[run]
        elif isinstance(run, (list, tuple, np.ndarray)):
            return np.concatenate([self.__runs[index] for index in run])
        else:
            raise IndexError(f'Invalid index type: {type(run)}')

    def __get(self, runs, columns, nooverlap: bool = False):
        runlist = []
        if isinstance(runs, int):
            runlist = [self.__runs[runs]]
        elif isinstance(runs, (list, tuple, np.ndarray)):
            runlist = [self.__runs[run] for run in runs]
        elif isinstance(runs, slice):
            runlist = self.__runs[runs]

        cutoffs = [run['time'][0] for run in runlist]
        if nooverlap:
            cutoffs = cutoffs[1:]
            cutoffs.append(runlist[-1]['time'][-1] + 1e-6)
        else:
            cutoffs = [np.max(cutoffs)+1e-6 for _ in cutoffs]

        if isinstance(columns, int):
            column = self.__columns[columns]
            return np.concatenate([run[column][run['time'] < cutoff] for run,cutoff in zip(runlist, cutoffs)], axis=0)
        elif isinstance(columns, str):
            return np.concatenate([run[columns] for run in runlist], axis=0)
        elif isinstance(columns, (list, tuple, np.ndarray)):
            return np.concatenate([run[list(columns)] for run in runlist], axis=0)
        elif isinstance(columns, slice):
            start = index.start if index.start is not None else 0
            stop = index.stop if index.stop is not None else len(self.__columns)
            step = index.step if index.step is not None else 1
            return np.concatenate([run[self.__columns[start:stop:step]] for run in runlist], axis=0)

    def get(self, index, index2 = None, nooverlap: bool = False):
        """
        Return the data from the specified column for all, or a specific run.

        Parameters
        ----------
        index : int | str | list | tuple | slice | np.ndarray
            Index or name of the column to be retrieved for all runs.
            When used with the second parameter, this references
            the index of the run instead and should be an int.

        index2 : int | str | list | tuple | slice | np.ndarray
            Index or name of the column to be retrieved
            from the run specified by the first index parameter.

        nooverlap : bool
            If True, assumes a succession of restarts and remove overlapping
            data points. The most recent run overwrites the overlapping data
            from the previous ones.
            Be careful, this means that the data will only go as far as
            the latest run goes. If a more recent run went less far
            than previous ones, the longer ones will not show up.

        Returns
        -------
        The data from the specified column, for all runs or the
        specified one.

        Raises
        ------
        RuntimeError
            If no file has been loaded in memory yet, either via the
            constructor or the readfile method.
        IndexError
            If the index type is invalid.
        """
        self.__checkloaded()
        if index is None:
            raise IndexError(f'Primary index cannot be None')
        elif index2 is None:
            if not isinstance(index, (int, str, list, tuple, np.ndarray, slice)):
                raise IndexError(f'Invalid index type: {type(index)}')
        else:
            if not isinstance(index, (int, list, tuple, np.ndarray, slice)) or not isinstance(index2, (int, str, list, tuple, np.ndarray, slice)):
                raise IndexError(f'Invalid index type: {type(index)} and {type(index2)}')

        # If index is int or str, retrieve data from all runs for corresponding columns
        if index2 is None:
            return self.__get(slice(None), index, nooverlap)
        # If index2 is used, retrieve column from specific run
        else:
            return self.__get(index, index2, nooverlap)

    def readfile(self, filename) -> None:
        """
        Reads the specified dat file and loads the data in memory.

        Parameters
        ----------
        filename
            The path to the dat file.
        """

        with open(filename, 'r') as f:
            l = f.readline().strip()

        # Read column headers. Strip the number at the beginning and save the name
        offset = 26
        width = 25
        self.__columns = []
        self.__runs = []
        # First column begins with a #
        self.__columns.append(re.sub(r'^\d+\s*', '', l[1:width]).strip())

        # Read the other columns
        while (offset + width < len(l)):
            column = re.sub(r'^\d+\s*', '', l[offset:offset+width].strip()).strip()
            self.__columns.append(column)
            offset += width+1

        # Read last one to avoid indexing out of bounds
        column = re.sub(r'^\d+\s*', '', l[offset:].strip()).strip()
        if len(column) > 0:
            self.__columns.append(column)

        with open(filename, 'r') as f:
            nline = 0
            run_starts = None
            for line in f:
                nline += 1
                
                if (line.strip().startswith('#')):
                    if run_starts is not None:
                        # Parse file between two headers (one FLASH run)
                        if (nline-run_starts-1) > 0:
                            data = np.genfromtxt(
                                fname=filename,
                                dtype=None,
                                names=self.__columns,
                                encoding='ascii',
                                skip_header=run_starts,
                                max_rows=nline-run_starts-1
                            )
                            self.__runs.append(np.atleast_1d(data))

                        run_starts = nline
                    else:
                        run_starts = nline

            # Parse last run, from header to EOF
            if run_starts is not None:
                if nline > run_starts:
                    data = np.genfromtxt(
                        fname=filename,
                        names=self.__columns,
                        dtype=None,
                        encoding='ascii',
                        skip_header=run_starts
                    )
                    self.__runs.append(data)

                    # Update column names to match python compliants one (_ instead of spaces and - etc)
                    self.__columns = self.__runs[0].dtype.names

            if len(self.__runs) > 0:
                self.__loaded = True
            else:
                # Nothing was found in the dat file
                # TODO Print some warning
                self.__loaded = False
