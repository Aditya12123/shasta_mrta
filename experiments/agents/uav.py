import numpy as np

from shasta.actor import BaseActor
import copy


class UaV(BaseActor):
    """This the base class for single UGV robot"""

    def __init__(self, config=None):
        # Platoon properties
        self.vehicle_id = 0
        self.platoon_id = 0
        self.type = 'uav'

        # Properties UAV
        self.init_pos = None
        self.init_orientation = None
        self.current_pos = copy.deepcopy(self.init_pos)
        self.desired_pos = copy.deepcopy(self.init_pos)

        # Extra parameters
        self.idle = True
        self.ammo = 100
        self.battery = 100
        self.functional = True
        self.speed = 2.5
        self.search_speed = 0.25

        return None

    def load_asset(self):
        """Initial step of objects and constraints"""
        # Initial pos and orientation
        if self.init_orientation is None:
            self.init_orientation = self.physics_client.getQuaternionFromEuler(
                [0, 0, np.pi / 2]
            )

        # Load the mesh
        path = '/'.join(['./assets', 'vehicles', 'arial_vehicle_abstract.urdf'])
        self.object = self.physics_client.loadURDF(
            path,
            self.init_pos,
            self.init_orientation,
            flags=self.physics_client.URDF_USE_MATERIAL_COLORS_FROM_MTL,
        )
        # Constraint
        self.constraint = self.physics_client.createConstraint(
            self.object,
            -1,
            -1,
            -1,
            self.physics_client.JOINT_FIXED,
            [0, 0, 0],
            [0, 0, 0],
            self.init_pos,
        )

        # Change color depending on team type
        self.physics_client.changeVisualShape(self.object, -1, rgbaColor=[0, 0, 1, 1])
        return None

    def get_pos_and_orientation(self):
        """Returns the position and orientation (as Yaw angle) of the robot."""
        pos, rot = self.physics_client.getBasePositionAndOrientation(self.object)
        euler = self.physics_client.getEulerFromQuaternion(rot)
        return np.array(pos), euler

    def reset(self):
        """Moves the robot back to its initial position"""
        self.physics_client.changeConstraint(self.constraint, self.init_pos)
        self.current_pos = copy.deepcopy(self.init_pos)
        self.desired_pos = copy.deepcopy(self.init_pos)
        return None

    def get_observation(self):
        pos, orientation = self.get_pos_and_orientation()
        return pos

    def apply_action(self, position):
        """This function moves the vehicles to given position

        Parameters
        ----------
        position : array
            The position to which the vehicle should be moved.
        """
        self.current_pos, _ = self.get_pos_and_orientation()
        position[2] = 10.0
        self.physics_client.changeConstraint(self.constraint, position)
        return None

    def destroy(self):
        self.physics_client.removeBody(self.object)
