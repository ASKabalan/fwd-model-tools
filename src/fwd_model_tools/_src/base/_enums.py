from enum import Enum, auto


class FieldStatus(str, Enum):
    """Lifecycle state for 3D volumetric fields."""

    UNKNOWN = auto()
    INITIAL_FIELD = auto()
    LPT1 = auto()
    LPT2 = auto()
    DENSITY_FIELD = auto()
    PARTICLES = auto()
    PROJECTED_DENSITY = auto()
    LIGHTCONE = auto()
    KAPPA = auto()
    GAMMA = auto()


class PhysicalUnit(str, Enum):
    INVALID_UNIT = auto()


class PositionUnit(str, Enum):
    GRID_RELATIVE = auto()
    GRID_ABSOLUTE = auto()
    MPC_H = auto()


class DensityUnit(str, Enum):
    OVERDENSITY = auto()
    DENSITY = auto()
    COUNTS = auto()
    MSUN_H_PER_MPC3 = auto()


class ConvergenceUnit(str, Enum):
    DIMENSIONLESS = auto()
    EFFECTIVE_DENSITY = auto()
