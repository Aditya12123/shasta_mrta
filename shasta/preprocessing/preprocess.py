import os
from pathlib import Path
import shutil


import networkx as nx
import osmnx as ox


class PreProcess:
    def __init__(self, osm_map_path=None, blender_path=None) -> None:
        self.osm_map_path = osm_map_path
        self.blender_path = blender_path

    def _save_building_info(self):
        raise NotImplementedError

    def _save_graph_info(self):
        raise NotImplementedError

    def run_processing():

        # Create the 3D world from the map.osm file
        os.system(
            "java -jar OSM2World.jar --config texture_config.properties -i map.osm -o map.obj"  # noqa
        )

        # Save OSMNX graph
        G = ox.graph_from_xml('map.osm', simplify=True, bidirectional='walk')
        G = nx.convert_node_labels_to_integers(G)

        # Use the blender to bake the texture
        blender_path = "/Applications/blender.app/Contents/MacOS/blender"
        os.system(blender_path + " --background  --python bake_texture.py")

        # Tidy up things and move the files to respective folders
        print('-' * 32)
        print('Preprocessing completed successfully')
        print('-' * 32)

        name = input("Please provide a name for the new asset: ")

        # Create a directory
        parent_folder = name
        meshes_folder = name + '/meshes'
        Path(parent_folder).mkdir(parents=True, exist_ok=True)
        Path(meshes_folder).mkdir(parents=True, exist_ok=True)

        # Move map.osm and coordinates inside the parent folder
        def move_files(file_name, directory):
            try:
                shutil.copy(file_name, directory + '/' + file_name)
            except FileNotFoundError:
                pass

        files = [
            'map.osm',
            'coordinates.csv',
            'environment.urdf',
            'environment_collision_free.urdf',
        ]
        for file in files:
            move_files(file, parent_folder)

        files = ['map.obj', 'map.png', 'map.mtl']
        for file in files:
            move_files(file, meshes_folder)

        # Finaly move the whole folder to data folder
        shutil.move(name, '../data/assets/')


if __name__ == "__main__":
    preprocess = PreProcess()
    preprocess.run_processing()
