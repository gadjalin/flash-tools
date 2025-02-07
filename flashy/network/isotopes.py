import numpy as np
import re

# Dictionary mapping element symbols and names to atomic numbers
_ELEMENTS = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20,
    "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30,
    "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36, "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40,
    "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
    "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57, "Ce": 58, "Pr": 59, "Nd": 60,
    "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70,
    "Lu": 71, "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80,
    "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85, "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90,
    "Pa": 91, "U": 92, "Np": 93, "Pu": 94, "Am": 95, "Cm": 96, "Bk": 97, "Cf": 98, "Es": 99, "Fm": 100,
    "Md": 101, "No": 102, "Lr": 103, "Rf": 104, "Db": 105, "Sg": 106, "Bh": 107, "Hs": 108, "Mt": 109,
    "Ds": 110, "Rg": 111, "Cn": 112, "Nh": 113, "Fl": 114, "Mc": 115, "Lv": 116, "Ts": 117, "Og": 118,
    "Hydrogen": 1, "Helium": 2, "Lithium": 3, "Beryllium": 4, "Boron": 5, "Carbon": 6, "Nitrogen": 7,
    "Oxygen": 8, "Fluorine": 9, "Neon": 10, "Sodium": 11, "Magnesium": 12, "Aluminium": 13, "Silicon": 14,
    "Phosphorus": 15, "Sulfur": 16, "Chlorine": 17, "Argon": 18, "Potassium": 19, "Calcium": 20,
    "Scandium": 21, "Titanium": 22, "Vanadium": 23, "Chromium": 24, "Manganese": 25, "Iron": 26,
    "Cobalt": 27, "Nickel": 28, "Copper": 29, "Zinc": 30, "Gallium": 31, "Germanium": 32, "Arsenic": 33,
    "Selenium": 34, "Bromine": 35, "Krypton": 36, "Rubidium": 37, "Strontium": 38, "Yttrium": 39,
    "Zirconium": 40, "Niobium": 41, "Molybdenum": 42, "Technetium": 43, "Ruthenium": 44, "Rhodium": 45,
    "Palladium": 46, "Silver": 47, "Cadmium": 48, "Indium": 49, "Tin": 50, "Antimony": 51, "Tellurium": 52,
    "Iodine": 53, "Xenon": 54, "Caesium": 55, "Barium": 56, "Lanthanum": 57, "Cerium": 58, "Praseodymium": 59,
    "Neodymium": 60, "Promethium": 61, "Samarium": 62, "Europium": 63, "Gadolinium": 64, "Terbium": 65,
    "Dysprosium": 66, "Holmium": 67, "Erbium": 68, "Thulium": 69, "Ytterbium": 70, "Lutetium": 71,
    "Hafnium": 72, "Tantalum": 73, "Tungsten": 74, "Rhenium": 75, "Osmium": 76, "Iridium": 77, "Platinum": 78,
    "Gold": 79, "Mercury": 80, "Thallium": 81, "Lead": 82, "Bismuth": 83, "Polonium": 84, "Astatine": 85,
    "Radon": 86, "Francium": 87, "Radium": 88, "Actinium": 89, "Thorium": 90, "Protactinium": 91,
    "Uranium": 92, "Neptunium": 93, "Plutonium": 94, "Americium": 95, "Curium": 96, "Berkelium": 97,
    "Californium": 98, "Einsteinium": 99, "Fermium": 100, "Mendelevium": 101, "Nobelium": 102, "Lawrencium": 103,
    "Rutherfordium": 104, "Dubnium": 105, "Seaborgium": 106, "Bohrium": 107, "Hassium": 108, "Meitnerium": 109,
    "Darmstadtium": 110, "Roentgenium": 111, "Copernicium": 112, "Nihonium": 113, "Flerovium": 114,
    "Moscovium": 115, "Livermorium": 116, "Tennessine": 117, "Oganesson": 118
}

def get_charge(element: str) -> int:
    """
    Returns the atomic number of an element given its symbol or full name (case-insensitive).

    Arguments
    ---
    element : str
        The symbol or full name of the element (case-insensitive).

    Returns
    ---
    Z : int
        The atomic number of the element.

    Raises
    ---
    ValueError
        If the element symbol or name is not found.
    """

    # Standardise input to title case
    element = element.strip().title()

    if element in _ELEMENTS:
        return _ELEMENTS[element]
    else:
        raise ValueError(f'Unrecognised element: {element}')

def parse_isotope_id(isotope: str) -> tuple[int, int]:
    """
    Return the atomic number and weight of the specified isotope.

    Arguments
    ---
    isotope : str
        The isotope symbol in the form e.g. Ni56.
        The string must start with the symbol or name of the element, and the full or shorten atomic weight of the isotope.
        For example, Pb07 must be parsed as Lead 207 (82, 207).

    Returns
    ---
    Z, A : tuple[int, int]
        A tuple with the atomic number first, and weight second.

    Raises
    ---
    ValueError
        If the isotope string cannot be parsed;
    """

    match = re.match(r'^([^\d]*)(.*)$', isotope)
    if match is None:
        raise ValueError(f'Unrecognised isotope: {isotope}')
    
    name = match[1]
    weight = match[2]
    Z = get_charge(name)
    
    if isotope.lower() == 'h': #Hydrogen 1 may omit atomic weight
        return 1, 1

    A_parsed = int(weight)
    if (A_parsed < 100):
        if (Z < 42): # Hydrogen to Niobium, A < 100
            A = A_parsed
        elif (Z >= 42 and Z <= 47): # Molybdenum to Silver, A ~ 100
            if (A_parsed > 80):
                A = A_parsed
            else:
                A = 100 + A_parsed
        elif (Z > 47 and Z < 78): # Cadmium to Iridium, 100 < A < 200
            A = 100 + A_parsed
        elif (Z >= 78 and Z <= 83): # Paladium to Bismuth, A ~ 200
            if (A_parsed > 80):
                A = 100 + A_parsed
            else:
                A = 200 + A_parsed
        elif (Z > 83): # Over Bismuth, A > 200
            A = 200 + A_parsed
    else:
        A = A_parsed

    return Z, A
