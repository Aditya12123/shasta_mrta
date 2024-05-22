import os
import psutil
import signal

import pybullet as p
from pybullet_utils import bullet_client as bc

from .world import World
from .map import Map

from .utils import get_initial_positions


def kill_all_servers():
    """Kill all PIDs that start with Carla"""
    processes = [p for p in psutil.process_iter() if "carla" in p.name().lower()]
    for process in processes:
        os.kill(process.pid, signal.SIGKILL)


class ShastaCore:
    """
    Class responsible of handling all the different CARLA functionalities,
    such as server-client connecting, actor spawning,
    and getting the sensors data.
    """

    def __init__(self, config, actor_groups: dict = None):
        """Initialize the server and client"""
        self.config = config
        self.actor_groups = actor_groups

        # Verify if the actor groups is a dictionary
        if not isinstance(self.actor_groups, dict):
            raise TypeError('Actor groups should be of type dict')

        # Setup world and map
        self.world = World(config)
        self.map = Map()

        self.init_server()
        self._setup_physics_client()

    def _setup_physics_client(self):
        """Setup the physics client

        Returns
        -------
        None
        """
        # Usage mode
        if self.config['headless']:
            self.physics_client = bc.BulletClient(connection_mode=p.DIRECT)
        else:
            options = '--background_color_red=0.85 --background_color_green=0.85 --background_color_blue=0.85'  # noqa
            self.physics_client = bc.BulletClient(
                connection_mode=p.GUI, options=options
            )

            # Set the camera parameters
            self.camer_distance = 150.0
            self.camera_yaw = 0.0
            self.camera_pitch = -89.999
            self.camera_target_position = [0, 30, 0]
            self.physics_client.resetDebugVisualizerCamera(
                cameraDistance=self.camer_distance,
                cameraYaw=self.camera_yaw,
                cameraPitch=self.camera_pitch,
                cameraTargetPosition=self.camera_target_position,
            )

            self.physics_client.configureDebugVisualizer(
                self.physics_client.COV_ENABLE_GUI, 0
            )

        # Set gravity
        self.physics_client.setGravity(0, 0, -9.81)

        # Set parameters for simulation
        self.physics_client.setPhysicsEngineParameter(
            fixedTimeStep=self.config['time_step'] / 10,
            numSubSteps=1,
            numSolverIterations=5,
        )

        # Inject physics client
        if self.world.physics_client is None:
            self.world.physics_client = self.physics_client

        return None

    def get_physics_client(self):
        """Ge the physics client

        Returns
        -------
        object
            The bullet physics client
        """
        return self.physics_client

    def init_server(self):
        """Start a server on a random port"""
        pass

    def setup_experiment(self, experiment_config):
        """Initialize the hero and sensors"""

        # Load the environment and setup the map
        self.map.setup(experiment_config)
        read_path = self.map.asset_path + '/environment_collision_free.urdf'
        self.world.load_world_model(read_path)

        # Spawn the actors in the physics client
        self.spawn_actors()

    def get_world(self):
        """Get the World object from the simulation

        Returns
        -------
        object
            The world object
        """
        return self.world

    def get_map(self):
        """Get the Map object from the simulation

        Returns
        -------
        object
            The map object
        """
        return self.map

    def reset(self):
        """This function resets / spawns the hero vehicle and its sensors"""

        # Reset all the actors
        for group_id in self.actor_groups:
            # Check if the entry is a list or not
            if not isinstance(self.actor_groups[group_id], list):
                self.actor_groups[group_id] = [self.actor_groups[group_id]]

            for actor in self.actor_groups[group_id]:
                # Reset the actor and collect the observation
                actor.reset()

        return None

    def spawn_actors(self):
        """Spawns vehicles and walkers, also setting up the Traffic Manager and its parameters"""
        for group_id in self.actor_groups:
            # Check if the entry is a list or not
            if not isinstance(self.actor_groups[group_id], list):
                self.actor_groups[group_id] = [self.actor_groups[group_id]]

            # Spawn the actors
            spawn_point = self.map.get_cartesian_spawn_points()
            positions = get_initial_positions(
                spawn_point, 10, len(self.actor_groups[group_id])
            )
            for actor, position in zip(self.actor_groups[group_id], positions):
                if actor.init_pos is None:
                    actor.init_pos = position
                else:
                    actor.init_pos = self.map.convert_to_cartesian(actor.init_pos)

                self.world.spawn_actor(actor, position)

    def get_actor_groups(self):
        """Get the actor groups

        Returns
        -------
        dict
            The actor groups as a dict with group id as the key
            list of actors as the value
        """
        return self.actor_groups

    def get_actors_by_group_id(self, group_id):
        """Get a list of actor given by group id

        Parameters
        ----------
        group_id : int
            Group id to be returned

        Returns
        -------
        list
            A list of actor given the group id
        """
        return self.actor_groups[group_id]

    def tick(self):
        """Performs one tick of the simulation, moving all actors, and getting the sensor data"""
        observations = {}

        # Tick once the simulation
        self.physics_client.stepSimulation()

        # Collect the raw observation from all the actors in each actor group
        for group in self.actor_groups:
            obs_from_each_actor = [
                actor.get_observation() for actor in self.actor_groups[group]
            ]

            observations[group] = obs_from_each_actor

        return observations

    def close_simulation(self):
        """Close the simulation"""
        p.disconnect(self.physics_client._client)
