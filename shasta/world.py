import numpy as np


class World:
    def __init__(self, config):
        super(World, self).__init__()
        # Need to specify some parameters
        self.config = config
        self.actor_ids = []

        # Setup the phsysics client
        self.physics_client = None

        return None

    def load_world_model(self, read_path):
        """Load the URDF model

        Parameters
        ----------
        read_path : str
            The path to the URDF model
        """

        if self.physics_client is not None:
            self.physics_client.loadURDF(
                read_path,
                [0, 0, 0],
                self.physics_client.getQuaternionFromEuler([np.pi / 2, 0, 0]),
                flags=self.physics_client.URDF_USE_MATERIAL_COLORS_FROM_MTL,
                useFixedBase=True,
            )

    def change_camera_position(
        self,
        camera_distance=150.0,
        camera_yaw=0.0,
        camera_pitch=-89.999,
        camera_target_position=[0, 30, 0],
    ):
        """Change the world camera position

        Parameters
        ----------
        camera_distance : float, optional
            Camera distance from target, by default 150
        camera_yaw : float, optional
            Camera yaw, by default 0
        camera_pitch : float, optional
            Camera pitch, by default -89.999
        camera_target_position : list, optional
            Camera target position, by default [0, 30, 0]

        Returns
        -------
        None
        """
        self.camer_distance = camera_distance
        self.camera_yaw = camera_yaw
        self.camera_pitch = camera_pitch
        self.camera_target_position = camera_target_position

        self.physics_client.resetDebugVisualizerCamera(
            cameraDistance=camera_distance,
            cameraYaw=camera_yaw,
            cameraPitch=camera_pitch,
            cameraTargetPosition=camera_target_position,
        )
        return None

    def spawn_actor(self, actor, spawn_point):
        """Spawn the actor at a given point

        Parameters
        ----------
        actor : object
            An actor to spawm
        spawn_point : array
            Spawn point

        Returns
        -------
        None
        """
        # Check if the physic client is added to the actor
        if actor.physics_client is None:
            actor.physics_client = self.physics_client

        # Load the actor
        actor._load()
        self.actor_ids.append(actor.get_actor_id())
        return None
