from abc import ABCMeta, abstractmethod
from future.utils import with_metaclass


class BaseActor(with_metaclass(ABCMeta, object)):
    """This the base class for single UGV robot"""

    physics_client = None
    _loaded = False

    def __init__(self):
        self.states = {}
        self._actor_id = None
        self.init_pos = None
        self.init_orientation = None

    def _load(self):
        """Load object into pybullet and return list of loaded body ids."""
        if self._loaded:
            raise ValueError("Cannot load an actor multiple times.")
        self._loaded = True
        actor_ids = self.load_asset()

        if not isinstance(actor_ids, list):
            actor_ids = [actor_ids]

        self._actor_id = actor_ids[0]
        return actor_ids

    def get_actor_id(self):
        """Get the actor id

        Returns
        -------
        int
            The actor id in the simulation
        """
        return self._actor_id

    @abstractmethod
    def load_asset(self, *args, **kwargs):
        """Load the assests of the actor."""
        raise NotImplementedError

    @abstractmethod
    def reset(self, *args, **kwargs):
        """Reset the actor

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    @abstractmethod
    def get_observation(self, *args, **kwargs):
        """Get the observation from the actor.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    @abstractmethod
    def apply_action(self, *args, **kwargs):
        """Apply the action to the actor

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    @abstractmethod
    def destroy(self, *args, **kwargs):
        """Destroy the actor

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError
