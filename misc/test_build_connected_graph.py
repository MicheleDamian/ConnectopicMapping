import numpy
from scipy.sparse import csgraph

def build_connected_graph_1(similarities):

    """ Binary search a similarity threshold for each node such that the pairs
        of voxels' with similarities greater than the threshold form a
        connected graph.
    """

    num_voxels = similarities.shape[0]
    high_index = num_voxels - 1
    low_index = 0

    adjacency = []
    survival_ratio = -1

    while True:

        adjacency_temp = numpy.zeros(similarities.shape, dtype=bool)

        similarity_index = int((high_index + low_index) / 2)
        similarity_argsort = numpy.argsort(similarities)

        cutoff_index = similarity_argsort[:, similarity_index]
        cutoff_value = similarities[list(range(num_voxels)), cutoff_index]

        for i in range(num_voxels):
            adjacency_voxel = similarities[i, :] >= cutoff_value[i]
            adjacency_temp[i, :] += adjacency_voxel
            adjacency_temp[:, i] += adjacency_voxel

        # Find connected components
        num_components, _ = csgraph.connected_components(adjacency_temp, directed=False)

        print(low_index, high_index, similarity_index, cutoff_value, num_components)
        print(adjacency_temp)

        if num_components > 1:
            high_index = similarity_index - 1
        else:
            low_index = similarity_index + 1
            adjacency = adjacency_temp
            survival_ratio = 1.0 - similarity_index / num_voxels

        if high_index < low_index:
            break

    return adjacency, survival_ratio


DIM = 7
SEED = 0

numpy.random.seed(SEED)

similarities = numpy.random.rand(DIM, DIM)

for i in range(DIM):
    similarities[i, :] = similarities[:, i]
    similarities[i, i] = 1.0

print(similarities)

adj, sur_ratio = build_connected_graph_1(similarities)

print(adj)
print(sur_ratio)
