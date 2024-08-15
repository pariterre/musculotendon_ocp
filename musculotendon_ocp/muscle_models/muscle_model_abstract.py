from abc import ABC, abstractmethod

from casadi import MX


class MuscleModelAbstract(ABC):
    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        """
        Get the muscle name
        """
        return self._name

    @abstractmethod
    def normalize_muscle_length(self, muscle_fiber_length: MX) -> MX:
        """
        Compute the normalized muscle length

        Parameters
        ----------
        muscle_fiber_length: MX
            The muscle length

        Returns
        -------
        MX
            The normalized muscle length
        """

    @abstractmethod
    def normalize_muscle_velocity(self, muscle_fiber_velocity: MX) -> MX:
        """
        Compute the normalized muscle velocity

        Parameters
        ----------
        muscle_fiber_velocity: MX
            The muscle velocity

        Returns
        -------
        MX
            The normalized muscle velocity
        """

    @abstractmethod
    def normalize_tendon_length(self, tendon_length: MX) -> MX:
        """
        Compute the normalized tendon length

        Parameters
        ----------
        tendon_length: MX
            The tendon length

        Returns
        -------
        MX
            The normalized tendon length
        """

    @abstractmethod
    def compute_muscle_force(self, activation: MX, muscle_fiber_length: MX, muscle_fiber_velocity: MX) -> MX:
        """
        Compute the muscle force

        Parameters
        ----------
        activation: MX
            The muscle activation
        muscle_fiber_length: MX
            The length of the muscle fibers
        muscle_fiber_velocity: MX
            The velocity of the muscle fibers

        Returns
        -------
        MX
            The muscle force corresponding to the given muscle activation, length and velocity
        """

    @abstractmethod
    def compute_tendon_force(self, tendon_length: MX) -> MX:
        """
        Compute the tendon force

        Parameters
        ----------
        tendon_length: MX
            The length of tendon unit

        Returns
        -------
        MX
            The tendon force corresponding to the given tendon length
        """
