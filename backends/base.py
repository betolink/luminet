"""Abstract base class for computational backends."""

from abc import ABC, abstractmethod
from typing import Tuple, Union
import numpy as np


class BaseBackend(ABC):
    """Abstract base class for computational backends.

    All backends (scipy, taichi, jax, etc.) must implement
    these methods to provide consistent interface.
    """

    def __init__(self):
        """Initialize backend."""
        pass

    @abstractmethod
    def calc_q(self, p: Union[float, np.ndarray], bh_mass: float) -> Union[float, np.ndarray]:
        """Convert periastron P to Q.

        Q = sqrt((P - 2M)(P + 6M))

        Args:
            p: Periastron distance
            bh_mass: Black hole mass

        Returns:
            Q value
        """
        pass

    @abstractmethod
    def calc_k_squared(self, p: Union[float, np.ndarray], bh_mass: float) -> Union[float, np.ndarray]:
        """Calculate squared modulus of elliptic integral.

        k^2 = m = (Q - P + 6M) / (2Q)

        Args:
            p: Periastron distance
            bh_mass: Black hole mass

        Returns:
            Squared modulus k^2
        """
        pass

    @abstractmethod
    def calc_sn(
        self,
        p: Union[float, np.ndarray],
        angle: Union[float, np.ndarray],
        bh_mass: float,
        incl: float,
        order: int = 0
    ) -> Union[float, np.ndarray]:
        """Calculate Jacobi elliptic function sn.

        Args:
            p: Periastron distance
            angle: Angle in black hole frame
            bh_mass: Black hole mass
            incl: Inclination angle
            order: Image order (0=direct, 1=ghost)

        Returns:
            sn value
        """
        pass

    @abstractmethod
    def calc_zeta_inf(self, p: Union[float, np.ndarray], bh_mass: float) -> Union[float, np.ndarray]:
        """Calculate zeta_infinity for elliptic integral.

        Args:
            p: Periastron distance
            bh_mass: Black hole mass

        Returns:
            zeta_inf value
        """
        pass

    @abstractmethod
    def periastron_cost(
        self,
        p: Union[float, np.ndarray],
        radius: Union[float, np.ndarray],
        angle: Union[float, np.ndarray],
        bh_mass: float,
        incl: float,
        order: int = 0
    ) -> Union[float, np.ndarray]:
        """Cost function for periastron optimization.

        Args:
            p: Periastron distance
            radius: Radius in black hole frame
            angle: Angle in black hole frame
            bh_mass: Black hole mass
            incl: Inclination angle
            order: Image order

        Returns:
            Cost value (should be 0 at solution)
        """
        pass

    @abstractmethod
    def solve_for_periastron(
        self,
        radius: float,
        incl: float,
        alpha: float,
        bh_mass: float,
        order: int = 0
    ) -> float:
        """Solve for periastron given black hole coordinates.

        Args:
            radius: Radius in black hole frame
            incl: Inclination angle
            alpha: Angle in black hole frame
            bh_mass: Black hole mass
            order: Image order

        Returns:
            Periastron distance
        """
        pass

    @abstractmethod
    def solve_for_impact_parameter(
        self,
        radius: float,
        incl: float,
        alpha: float,
        bh_mass: float,
        order: int = 0
    ) -> float:
        """Solve for impact parameter b.

        Args:
            radius: Radius in black hole frame
            incl: Inclination angle
            alpha: Angle in black hole frame
            bh_mass: Black hole mass
            order: Image order

        Returns:
            Impact parameter b
        """
        pass

    @abstractmethod
    def calc_redshift_factor(
        self,
        radius: Union[float, np.ndarray],
        angle: Union[float, np.ndarray],
        incl: float,
        bh_mass: float,
        b: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Calculate gravitational redshift factor (1+z).

        Args:
            radius: Radius in black hole frame
            angle: Angle in black hole frame
            incl: Inclination angle
            bh_mass: Black hole mass
            b: Impact parameter

        Returns:
            Redshift factor (1+z)
        """
        pass

    @abstractmethod
    def calc_flux_intrinsic_swarzschild(
        self,
        radius: Union[float, np.ndarray],
        acc: float,
        bh_mass: float
    ) -> Union[float, np.ndarray]:
        """Calculate intrinsic flux for Schwarzschild black hole.

        Args:
            radius: Radius in black hole frame
            acc: Accretion rate
            bh_mass: Black hole mass

        Returns:
            Intrinsic flux
        """
        pass

    @abstractmethod
    def calc_flux_observed(
        self,
        radius: Union[float, np.ndarray],
        acc: float,
        bh_mass: float,
        redshift_factor: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Calculate observed flux with redshift correction.

        Args:
            radius: Radius in black hole frame
            acc: Accretion rate
            bh_mass: Black hole mass
            redshift_factor: Redshift factor (1+z)

        Returns:
            Observed flux
        """
        pass

    @abstractmethod
    def get_backend_name(self) -> str:
        """Get the name of this backend."""
        pass

    @abstractmethod
    def supports_vectorization(self) -> bool:
        """Check if backend supports vectorized operations."""
        pass

    @abstractmethod
    def supports_gpu(self) -> bool:
        """Check if backend can run on GPU."""
        pass
