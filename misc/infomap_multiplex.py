#!/usr/bin/env python

""" Build a multiplex network of voxels correlations and find communities in
    the network.

"""

import sys

sys.path.append("/Users/michele/Development/infomap/examples/python/infomap/")
import infomap

import numpy
import json
import os
import multiprocessing
import networkx
from connectopic_mapping import utils
from scipy.sparse import csgraph
from matplotlib import pyplot

__author__ = "Michele Damian"
__email__ = "michele.damian@gmail.com"
__version__ = "0.3.0"


def build_sparse_graph(fingerprints, start_idx, end_idx, out_path):

    """ Compute distance matrix between pair of fingerprints for each non-ROI
        voxel.
    """

    num_voxels = fingerprints.shape[0]
    num_layers = fingerprints.shape[1]

    fname_similarity_multiplex = out_path + '/similarity_multiplex_{0}-{1}.npy'.format(start_idx, end_idx)

    if os.path.exists(fname_similarity_multiplex):
        similarity_multiplex = numpy.load(fname_similarity_multiplex)

    else:

        distance_multiplex = numpy.zeros((num_voxels, num_voxels, num_layers), dtype=numpy.float16)

        old_progress = -1

        for i in range(num_voxels):

            new_progress = int(100 * i / num_voxels)

            if new_progress > old_progress:
                print("\rCompute dense graph... {0}%".format(new_progress), end="", flush=True)
                old_progress = new_progress

            for j in range(num_voxels):

                coor_ij = fingerprints[[i, j], :]

                distance_pos = numpy.linalg.norm(1 - coor_ij, axis=0)
                distance_neg = numpy.linalg.norm(1 + coor_ij, axis=0)
                distances = numpy.stack((distance_pos, distance_neg), axis=0)

                distance_multiplex[i, j, :] = numpy.min(distances, axis=0)

        similarity_multiplex = 1.0 - distance_multiplex / 2.0

        del distance_multiplex

        numpy.save(fname_similarity_multiplex, similarity_multiplex)

    graph_sparse = []

    for l in range(num_layers):

        graph_sparse.append(networkx.Graph())

        adjacency, survival_ratio = build_connected_graph_1(similarity_multiplex[:, :, l])

        for i in range(num_voxels):
            for j in range(i+1, num_voxels):

                weight = float(similarity_multiplex[i, j, l])

                if adjacency[i, j]:
                    graph_sparse[l].add_edge(i, j, weight=weight)

        print("Sparse graph at layer {0}: {1} edges, {2} nodes, {3:.2f}% survival ratio, {4:.2f}% of edges"
              .format(start_idx + l,
                      graph_sparse[l].number_of_edges(),
                      graph_sparse[l].number_of_nodes(),
                      survival_ratio * 100,
                      (numpy.sum(adjacency, axis=None) - num_voxels) * 100 / (num_voxels * (num_voxels - 1))),
              flush=True)

    return graph_sparse


def build_connected_graph_0(similarities):

    """ Binary search a similarity threshold such that the pairs of voxels' with
        similarities greater than the threshold form a connected graph.
    """

    similarity_values = numpy.sort(similarities, axis=None)
    high_index = similarities.shape[0]**2 - 1
    low_index = 0
    max_threshold = similarity_values[0]

    while True:

        similarity_index = int((high_index + low_index) / 2)
        similarity_threshold = similarity_values[similarity_index]

        # Transform similarity matrix into a connected graph
        adjacency = similarities >= similarity_threshold

        # Find connected components
        num_components, _ = csgraph.connected_components(adjacency, directed=False)

        if num_components > 1:
            high_index = similarity_index - 1
        else:
            low_index = similarity_index + 1
            max_threshold = max(max_threshold, similarity_threshold)

        if high_index < low_index:
            break

    return max_threshold


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

        if num_components > 1:
            high_index = similarity_index - 1
        else:
            low_index = similarity_index + 1
            adjacency = adjacency_temp
            survival_ratio = 1.0 - similarity_index / num_voxels

        if high_index < low_index:
            break

    return adjacency, survival_ratio


# READ INPUT

with open('../scripts/config.json') as config_file:
    config = json.load(config_file)

subject = config["subject"]
session = config["session"]
scans = config["scans"]
hemisphere = config["hemisphere"]
atlas_name = config["atlas_name"]
roi_name = config["roi_name"]

image_path = config["nifti_dir_path"]

image_path_0 = image_path + \
               '/rfMRI_{1}_{2}_hp2000_clean.nii.gz' \
               .format(subject, session, scans[0])

image_path_1 = image_path + \
               '/rfMRI_{1}_{2}_hp2000_clean.nii.gz' \
               .format(subject, session, scans[1])

out_path = config["out_path"] + \
           '/rfMRI_{0}_{1}_{2}'.format(subject, session, hemisphere)

num_layers = 50


# COMPUTE SVD

fname_fingerprints = out_path + '/fingerprints_{0}.npy'.format(num_layers)

if os.path.exists(fname_fingerprints):
    fingerprints = numpy.load(fname_fingerprints)

else:

    print("Loading brain and ROI masks from atlas...", end="", flush=True)

    brain_mask, roi_mask = utils.load_masks(atlas_name, roi_name, hemisphere)

    print("\rLoading brain and ROI masks from atlas... Done!", flush=True)

    #
    # Load Nifti images, smooth with FWHM=6, compute % temporal change
    #

    print("Loading Nifti images (1/2)...", end="", flush=True)

    data_info_0 = utils.normalize_nifti_image(image_path_0, fwhm=6)

    print("\rLoading Nifti images (2/2)...", end="", flush=True)

    data_info_1 = utils.normalize_nifti_image(image_path_1, fwhm=6)

    print("\rLoading Nifti images... Done!", flush=True)

    #
    # Concatenate data from the two scans along the temporal axis
    #

    print("Concatenating Nifti images...", end="", flush=True)

    brain_mask, roi_mask, data = utils.concatenate_data(brain_mask, roi_mask,
                                                        *data_info_0, *data_info_1)

    # Dereference unnecessary data
    del data_info_0, data_info_1

    print("\rConcatenating Nifti images... Done!", flush=True)

    num_fingerprints = numpy.sum(roi_mask, axis=None)

    data_u, data_s, data_v = numpy.linalg.svd(data, full_matrices=False)

    numpy.save(out_path + '/data_s.npy', data_s)

    data_s = data_s[:num_layers]
    data_v = data_v[:num_layers, :num_fingerprints]

    fingerprints = numpy.dot(numpy.diag(data_s), data_v).T

    numpy.save(fname_fingerprints, fingerprints)


# BUILD SPARSE GRAPH

print("Build sparse graph...", flush=True)

fname_graph_sparse = out_path + "/graph_sparse.npy"

if os.path.exists(fname_graph_sparse):
    graph_sparse_numpy = numpy.load(fname_graph_sparse)

    graph_sparse = [networkx.Graph() for i in range(num_layers)]

    for i in range(graph_sparse_numpy.shape[0]):
        layer, edge_i, edge_j, weight = graph_sparse_numpy[i, :]
        layer = int(layer)
        edge_i = int(edge_i)
        edge_j = int(edge_j)
        graph_sparse[layer].add_edge(edge_i, edge_j, weight=weight)

    del graph_sparse_numpy

else:

    num_processes = 3
    pool = multiprocessing.Pool(num_processes)

    graph_sparse = [networkx.Graph()] * num_layers

    num_layers_process = int(num_layers / num_processes)
    pool_idx = [numpy.arange(i*num_layers_process, (i + 1)*num_layers_process)
                for i in range(num_processes)]
    pool_idx[-1] = numpy.append(pool_idx[-1], range(pool_idx[-1][-1] + 1, num_layers))
    starmap_input = [(fingerprints[:, idx], idx[0], idx[-1], out_path) for idx in pool_idx]

    graph_chunks = pool.starmap(build_sparse_graph, starmap_input)

    for p in range(num_processes):
        start_idx = pool_idx[p][0]
        for l in pool_idx[p]:
            graph_sparse[l] = graph_chunks[p][l - start_idx]

    # Save graph

    graph_sparse_numpy = []

    for l in range(num_layers):
        for edge in graph_sparse[l].edges_iter(data='weight'):
            edge_i, edge_j, weight = edge
            graph_sparse_numpy.append([l, edge_i, edge_j, weight])

    graph_sparse_numpy = numpy.asarray(graph_sparse_numpy)

    numpy.save(fname_graph_sparse, graph_sparse_numpy)

    del graph_chunks, graph_sparse_numpy

print("Build sparse graph... Done!", flush=True)


# INFOMAP

print("Build infomap model...", end="", flush=True)

infomap_model = infomap.MemInfomap(#"--overlapping "
                                   "--zero-based-numbering "
                                   "--seed 42 "
                                   #"--two-level"
)

for l in range(num_layers):

    progress = int(100 * l / num_layers)

    print("\rBuild infomap model... {0}%".format(progress), end="", flush=True)

    for edge in networkx.get_edge_attributes(graph_sparse[l], 'weight').items():
        i, j = edge[0]
        weight = edge[1]
        infomap_model.addMultiplexLink(l, i, l, j, weight)

print("\rBuild infomap model... Done!", flush=True)

print("# edges = {0}, # nodes = {1}".format(graph_sparse[0].number_of_edges(), graph_sparse[0].number_of_nodes()))

print("Find communities...", flush=True)
infomap_model.run()
print("Find communities... Done!", flush=True)

print("Found {0} top modules with codelength: {1}".format(infomap_model.tree.numTopModules(), infomap_model.tree.codelength()))


# PRINT GRAPH

with open(out_path + '/out.csv', 'w') as fout:

    fout.write("state ID,physical ID,module ID\n")

    communities = {}
    for node in infomap_model.tree.leafIter(0):
        fout.write("{0},{1},{2}\n".format(node.stateIndex, node.physIndex, node.moduleIndex()))
        if communities.get(node.physIndex) is None:
            communities[node.physIndex] = []
        communities[node.physIndex].append(node.moduleIndex())

    print("# nodes = {0}".format(id))


num_communities = max(communities.values()) + 1
print("# communities: {0}".format(num_communities))

networkx.set_node_attributes(graph_sparse[0], 'community', communities)

cmap = pyplot.get_cmap('gist_ncar')

weights = [e[2]['weight'] for e in graph_sparse[0].edges(data=True)]

pos = networkx.spring_layout(graph_sparse[0])
networkx.draw_networkx_edges(graph_sparse[0], pos, width=10*weights)

node_colors = [cmap(val[0] / num_communities) if len(val) == 1 else (0.5, 0.5, 0.5, 0.1)
               for val in list(communities.values())]
node_collection = networkx.draw_networkx_nodes(graph_sparse[0],
                                               pos=pos,
                                               node_size=100,
                                               node_color=node_colors)

pyplot.axis('off')
pyplot.show()
