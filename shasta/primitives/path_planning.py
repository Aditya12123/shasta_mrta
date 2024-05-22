import numpy as np
import networkx as nx
import osmnx as ox


class PathPlanning(object):
    """Path planner based on the skeleton of the image.
    Generates a spline path
    """

    def __init__(self, env_map):
        self.map = env_map
        self.A, self.G = self.map.get_affine_transformation_and_graph()
        return None

    def linear_refine_implicit(self, x, n):
        """Given a 2D ndarray (npt, m) of npt coordinates in m dimension,
        insert 2**(n-1) additional points on each trajectory segment
        Returns an (npt*2**(n-1), m) ndarray

        Parameters
        ----------
        x : array
            A 2D input array
        n : int
            Number of intermediate points to insert between two consecutive points in x

        Returns
        -------
        array
            An array with interploated points

        Raises
        ------
        NotImplementedError
            The functions is not implemented for 3D or higher dimensions
        ValueError
            Number of intermediate points should be greated than zero
        """
        if n > 1:
            m = 0.5 * (x[:-1] + x[1:])
            if x.ndim == 2:
                msize = (x.shape[0] + m.shape[0], x.shape[1])
            else:
                raise NotImplementedError

            x_new = np.empty(msize, dtype=x.dtype)
            x_new[0::2] = x
            x_new[1::2] = m
            return self.linear_refine_implicit(x_new, n - 1)
        elif n == 1:
            return x
        else:
            raise ValueError

    def find_path(self, start, end, n_splits=1):
        """Finds a path between start and end using path graph

        Parameters
        ----------
        start : array
            A cartesian co-ordinate specifying the start position
        end : array
            A node ID specifying the end position
        n_splits : int, optional
            Number of splits in refining the path points, by default 1

        Returns
        -------
        path_points : array
            A refined path points in pybullet cartesian co-ordinate system
        """
        x = []
        y = []

        # Convert the node index to node ID
        # TODO: Very dirty way to implement
        # TODO: Verify the implementation of nearest nodes
        if not isinstance(start, (int, np.int64)):
            start_lat_lon = self.map.convert_to_lat_lon(start)
            start = ox.distance.nearest_nodes(
                self.G, X=start_lat_lon[1], Y=start_lat_lon[0]
            )
        if not isinstance(end, (int, np.int64)):
            end_lat_lon = self.map.convert_to_lat_lon(end)
            end = ox.distance.nearest_nodes(self.G, X=end_lat_lon[1], Y=end_lat_lon[0])

        route = nx.shortest_path(self.G, start, end, weight='length')
        for u, v in zip(route[:-1], route[1:]):
            # if there are parallel edges, select the shortest in length
            data = min(self.G.get_edge_data(u, v).values(), key=lambda d: d["length"])
            if "geometry" in data:
                # if geometry attribute exists, add all its coords to list
                xs, ys = data["geometry"].xy
                x.extend(xs)
                y.extend(ys)
            else:
                # otherwise, the edge is a straight line from node to node
                x.extend((self.G.nodes[u]["x"], self.G.nodes[v]["x"]))
                y.extend((self.G.nodes[u]["y"], self.G.nodes[v]["y"]))

        # Add more path points for smooth travel
        lat_lon = np.array((x, y)).T
        refined_points = self.linear_refine_implicit(lat_lon, n=n_splits)
        refined_points = np.hstack(
            (refined_points, np.ones((refined_points.shape[0], 1)))
        )

        # Exchange x and y as they are reversed in pybullet
        refined_points[:, [1, 0]] = refined_points[:, [0, 1]]

        # Convert the lat lon to pybullet co-ordinates
        path_points = np.dot(refined_points, self.A)

        return path_points

    def execute(self):
        raise NotImplementedError
