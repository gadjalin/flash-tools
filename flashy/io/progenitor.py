import numpy as np
from collections.abc import Mapping, Sequence


class progenitor(object):
    """
    Store a FLASH progenitor.

    This class can read progenitor profiles
    from a FLASH progenitor file, or take data from memory.
    The profiles of each variable can then be used,
    and the progenitor can be saved to a file.
    """

    # Private fields
    __comment: str
    __columns: np.ndarray
    __data: np.ndarray
    __loaded: bool

    # Constructor
    def __init__(self):
        self.__comment = ''
        self.__columns = np.emty(0)
        self.__data = np.empty(0)
        self.__loaded = False

    # Alternative constructors
    @classmethod
    def fromfile(cls, filename):
        """
        Initialise progenitor object from a FLASH progenitor profile

        TODO: Find a way to provide the class with custom
        reader functions for different file formats than FLASH
        e.g. MESA, Woosley&Heger, etc

        Parameters
        ----------
        filename
            The path to the file to read the data from.
        """
        prog = cls()
        prog.readfile(filename)
        return prog

    @classmethod
    def fromdata(cls, r: Sequence[float], data: Mapping[str, Sequence[float]], comment: str = ""):
        prog = cls()
        prog.setdata(r, data, comment)
        return prog

    def isloaded(self) -> bool:
        return self.__loaded

    def __checkloaded(self) -> None:
        if not self.__loaded:
            raise RuntimeError('No progenitor file has been loaded yet!')

    def columns(self):
        """
        Get the name of the columns.
        """
        self.__checkloaded()
        return self.__columns

    def __getitem__(self, index: int | str):
        return self.get(index)

    def get(self, index: int | str):
        """
        Get the data in the specified column.

        Parameters
        ----------
        index : int | str
            Index of the column, either by its name, or numerical index.

        Returns
        -------
        The data from the specified column.

        Raises
        ------
        RuntimeError
            If no data has been loaded in memory yet.
        IndexError
            If the index type is invalid.
        """

        self.__checkloaded()
        if isinstance(index, int):
            return self.__data[self.__data.dtype.names[index]]
        elif isinstance(index, str):
            return self.__data[index]
        else:
            raise IndexError(f'Invalid index: {index}')

    def readfile(self, filename: str) -> None:
        """
        Read a progenitor file and loads the data in memory.

        Parameters
        ----------
        filename : str
            The path to the progenitor file.
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
            self.__columns = ['r']
            self.__columns += [f.readline().split()[0] for i in range(num_vars)]

        # Load columns
        self.__data = np.genfromtxt(
            fname=filename,
            skip_header=skip,
            names=self.__columns,
            dtype=None,
            encoding='ascii'
        )

        self.__loaded = True

    def setdata(r: Sequence[float], data: Mapping[str, Sequence[float]], comment: str = "") -> None:
        """
        Set the data describing the progenitor.
        
        Set the data for the progenitor from a list of mid-cell radii
        and a dictionary mapping the name of variables to their values
        for each given radius.

        Parameters
        ----------
        r : Sequence[float]
            Mid-cell radii of the progenitor profile in increasing order.
        data : Mapping[str, Sequence[float]]
            Dictionary with the name of the relevant variables as keys
            and their values at each given cell radius.
        comment : str
            An additional comment to be used when saving to a file.
        """
        self.__columns = ['r']
        self.__columns += list(data.keys())
        dtype = [(key, float) for key in data.keys()]
        self.__data = np.array(list(zip(r, *data.values)), dtype=dtype)
        self.__loaded = True

    def savefile(self, filename):
        """
        Save the progenitor profile to a new file.
    
        Parameters
        ----------
        filename
            The name of the file where to save the progenitor.

        Raises
        ------
        RuntimeError
            If no data has been loaded in memory yet.
        """
        self.__checkloaded()
        with open(filename, 'w') as f:
            print("#", self.__comment, file=f)
            print("number of variables =", len(self.__data) - 1, file=f)

            # Don't print the 'r' var name
            for var_name in self.__columns[1:]:
                print(var_name, file=f)
    
            for i in range(len(self.__data['r'])):
                print(self.__data['r'][i], *[self.__data[var_name][i] for var_name in self.__columns], file=f)


    # Although I don't know why you would do that
    def clear(self):
        """
        Restore the object to its default, unloaded state.
        
        Clear all data from memory, if any.
        """
        self.__comment = ''
        self.__columns = np.emty(0)
        self.__data = np.empty(0)
        self.__loaded = False
        