import numpy as np

from shasta.primitives import PathPlanning, Formation


class FormationWithPlanning:
    def __init__(self, env_map) -> None:
        self.planning = PathPlanning(env_map)
        self.formation = Formation()

        # Parameters
        self.path_points = None
        self.centroid = None

    def get_centroid(self, vehicles):
        """Get the centroid of the vehicles"""
        centroid = []
        if len(vehicles) > 1:
            for vehicle in vehicles:
                centroid.append(vehicle.current_pos)
            centroid = np.mean(np.asarray(centroid), axis=0)
        else:
            centroid = vehicles[0].current_pos
        return centroid

    def execute(self, vehicles, target_pos):

        # Get centroid and find the shortest path
        self.centroid = self.get_centroid(vehicles)

        # Find the path
        if self.path_points is None:
            self.path_points = self.planning.find_path(
                start=self.centroid, end=target_pos
            )

        # Start executing the action
        if len(self.path_points) > 0:
            self.next_pos = self.path_points[0]
            self.formation.execute(vehicles, self.next_pos, self.centroid, 'solid')

            # Update the path points
            self.update_path_points()

        return self.get_done_status()

    def update_path_points(self):
        distance = np.linalg.norm(self.centroid[0:2] - self.next_pos[0:2])

        if len(self.path_points) > 1 and distance < 0.5:
            self.path_points = np.delete(self.path_points, 0, 0)

    def get_done_status(self):
        if len(self.path_points) <= 1:
            return True
        else:
            return False
