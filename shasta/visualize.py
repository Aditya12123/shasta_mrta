import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox


def plot_nodes(config):
    """Visualize the node graph

    Parameters
    ----------
    config : yaml
        a yaml file providing the configuration
    """
    read_path = '/'.join(
        [config['urdf_data_path'], config['simulation']['map_to_use'], 'map.osm']
    )
    G = ox.graph_from_xml(read_path, simplify=True, bidirectional='walk')
    G = nx.convert_node_labels_to_integers(G)
    fig, ax = ox.plot_graph(G, show=False)
    for i, node in enumerate(G.nodes):
        ax.annotate(str(i), (G.nodes[i]['x'], G.nodes[i]['y']), c='w')
    plt.tight_layout()
    plt.show()
