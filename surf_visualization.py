import os
import nibabel
import numpy
from surfer import Brain
from surfer import io


def visualize(roi_mask, connectopies, mri_filename, hemisphere):

    # Set voxels values
    coords = numpy.where(roi_mask)
    data = numpy.zeros(roi_mask.shape)
    data[coords] = connectopies

    # Transform connectopic map into Nifti image
    nifti = nibabel.Nifti1Image(data, numpy.eye(4))
    nifti.header.set_data_dtype(numpy.float64)
    nifti.to_filename(mri_filename)

    # Transform volume (Nifti image) into surface
    reg_filename = os.path.join(os.environ["FREESURFER_HOME"],
                                "average/mni152.register.dat")
    surf_data = io.project_volume_data(mri_filename, hemisphere, reg_filename)

    min = numpy.min(surf_data)
    max = numpy.max(surf_data)

    # Visualize surface
    brain = Brain("fsaverage", hemisphere, "pial")
    brain.add_data(surf_data, min, max, colormap="terrain", smoothing_steps=0)
