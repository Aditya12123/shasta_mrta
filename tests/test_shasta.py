from shasta.actor import BaseActor
from shasta.assets import assets_root


class TestUxV(BaseActor):
    def __init__(self):
        super().__init__()

    def _load(self):
        path = '/'.join([assets_root, 'vehicles', 'arial_vehicle_abstract.urdf'])
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

    def reset(self):
        pass

    def get_observation(self):
        pass

    def apply_action(self, action):
        raise NotImplementedError

    def destroy(self):
        pass
