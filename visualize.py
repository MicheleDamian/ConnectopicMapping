import json
import numpy
from connectopic_mapping import utils
from matplotlib import pyplot

with open('config.json') as config_file:
    config = json.load(config_file)

subject = config["subject"]
session = config["session"]
scans = config["scans"]
hemisphere = config["hemisphere"]
atlas_name = config["atlas_name"]
roi_name = config["roi_name"]

out_path = config["out_path"] + \
           '/rfMRI_{0}_{1}_{2}'.format(subject, session, hemisphere)

brain_mask = numpy.load(out_path + "/brain_mask.npy")
roi_mask = numpy.load(out_path + "/roi_mask.npy")
connectopic_map = numpy.load(out_path + "/connectopic_map_{0}.npy".format(config["manifold_learning"]))

i_plot = 1

for config_figures in config['figures']:

    slice_indexes = [config_figures['axis_x'],
                     config_figures['axis_y'],
                     config_figures['axis_z']]

    if hemisphere == 'RH':
        slice_indexes[0] = brain_mask.shape[0] - slice_indexes[0]

    #
    # Display connectopy
    #
    fig = pyplot.figure(i_plot, tight_layout=True)
    utils.visualize_volume(-connectopic_map, brain_mask, roi_mask, slice_indexes,
                           low_percentile=1, high_percentile=99,
                           num_fig=fig,
                           title="Connectopies subject {0} {1}".format(subject, session),
                           margin=2,
                           legend_location=config_figures['legend_location'])

    i_plot += 1

pyplot.show()
