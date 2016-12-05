Notes
=====

The pipeline provides two variations from Haak and colleagues [1]_. In particular, this algorithm computes the fingerprints in a different way and it provides the option to use tSNE as manifold learning algorithm. The following sections describe in detail the two variations and present the results.

Fingerprinting
--------------

Haak and collegues [1]_ compute the fingerprints for each voxel in the ROI as the correlation between the time-serie represented by that voxel and the time-series obtained from a subspace of the voxels outside of the ROI. The procedure is the following: extract the OxT matrix (where O is the number of out-ROI voxels and T the length of the time-serie) from the fMRI image,reduce the out-ROI dimensionality by singular value decomposition (SVD) to obtain a new (T-1)-dimensional subspace from the O-dimensional space, compute the fingerprints as the Pearson's correlation coefficients between each in-ROI voxel and the T-1 "voxels" from the new subspace.

The goal of Haak and collegues is to reduce the amount of memory used and build "good" fingerprints. (The fingerprints are good if the similarity measure between the fingerprints of two in-ROI voxels is high when the two voxels have the same correlations with the out-ROI voxels, low when the correlations are different). However, when they reduce the space dimensionality, the correlations between in-ROI and out-ROI voxels are lost; the goodness of the fingerprints now depends on the orientation of the subspace relative to the original space. A more natural way to extract good fingerprints is to use the SVD's eigenvectors of the entire (in-ROI and out-ROI voxels) space. Since the right eigenvectors provide a mapping between the voxels and the desired subspace, it is possible to use the eta2 coefficients used by Haak and colleagues, this time on the mappings, to compare the voxels.

This implementation exploits this fact by the following steps: reduce the dimensionality of the N-dimensional space (where N is the number of all the voxels that represent the cortex) by SVD, consider the right eigenvectors associated with the in-ROI voxels as fingerprints. This doesn't require computing the Pearson's correlation coefficients making the pipeline faster.

Manifold learning
-----------------

The user can choose to use tSNE [2]_ as manifold learning algorithm—instead of spectral embedding or isomap. tSNE is conceived mainly as a visualization algorithm, nevertheless it provides as output clusters of data that are well separated from each other and can be used to find well-defined connectopies. The reason why it is not used as the default is because it is sensitive to noisy data. Despite tSNE consinstently groups the voxels in the same clusters, the position of the clusters on the output dimension changes; in other words, it can visualize well-defined connectopies, but it doesn't allow to analytically compare cross-sessions or cross-subjects results. An extension of this pipeline would be to automatically match clusters obtained from different sessions/subjects.

Results
-------

The results show equal or better cross-session reproducibility and present specific differences between opposite hemispheres and subjects. The mean normalized root mean squared error (nRMSE) for the first connectopy in Haak and collegues [1]_ using spectral embedding is *0.061* for the left and *0.067* for the right hemisphere (95% confidence interval of *[0.053, 0.070]* and *[0.056, 0.077]* respectively); using isomap is *0.039* for the left and *0.041* for the right hemisphere (95% confidence interval of *[0.035, 0.043]* and *[0.037, 0.046]* respectively).

Spectral embedding
~~~~~~~~~~~~~~~~~~

**Cross-session normalized root mean squared error for the left and right hemispheres.**

+---------+------------------------------------------------+
|         |                     nRMSE                      |
+ Subject +-----------------------+------------------------+
|         |    Left Hemisphere    |    Right Hemisphere    |
+=========+=======================+========================+
| 100307  | 0.02835               | 0.03866                |
+---------+-----------------------+------------------------+
| 103414  | 0.04054               | 0.05987                |
+---------+-----------------------+------------------------+
| 105115  | 0.02964               | 0.03264                |
+---------+-----------------------+------------------------+


**Cross-subject normalized root mean squared error for session 1 and session 2 and for the left and right hemispheres.**

+---------------+---------+------------------------------------------------+
|               |         |                     nRMSE                      |
| Subjects      | Session +-----------------------+------------------------+
|               |         |    Left Hemisphere    |    Right Hemisphere    |
+===============+=========+=======================+========================+
|               |       1 | 0.06109               | 0.06267                |
| 100307/103414 +---------+-----------------------+------------------------+
|               |       2 | 0.06161               | 0.07023                |
+---------------+---------+-----------------------+------------------------+
|               |       1 | 0.06101               | 0.05632                |
| 100307/105115 +---------+-----------------------+------------------------+
|               |       2 | 0.07289               | 0.06792                |
+---------------+---------+-----------------------+------------------------+
|               |       1 | 0.05811               | 0.06199                |
| 103414/105115 +---------+-----------------------+------------------------+
|               |       2 | 0.08603               | 0.10522                |
+---------------+---------+-----------------------+------------------------+


The following images show the connectopic mapping for subjects 100307, 103414 and 105115 from the `Human Connectome Project <https://www.humanconnectome.org>`_. The spatial distribution of the connectopies is very similar to Haak and colleagues' results.

**Connectopies of subject 100307 from three different planes orthogonal to the X, Y, and Z axes.**

+-----------+-----------------------------------------------------------------------------+-----------------------------------------------------------------------------+
|           |                             Left Hemisphire                                 |                             Right Hemisphire                                |
+-----------+-----------------------------------------------------------------------------+-----------------------------------------------------------------------------+
| Session 1 | .. image:: _static/Connectopies_subject_100307_REST1_planes_X25-Y55-Z69.png | .. image:: _static/Connectopies_subject_100307_REST1_planes_X66-Y55-Z69.png |
+-----------+-----------------------------------------------------------------------------+-----------------------------------------------------------------------------+
| Session 2 | .. image:: _static/Connectopies_subject_100307_REST2_planes_X25-Y55-Z69.png | .. image:: _static/Connectopies_subject_100307_REST2_planes_X66-Y55-Z69.png |
+-----------+-----------------------------------------------------------------------------+-----------------------------------------------------------------------------+

+-----------+-----------------------------------------------------------------------------+-----------------------------------------------------------------------------+
|           |                             Left Hemisphire                                 |                             Right Hemisphire                                |
+-----------+-----------------------------------------------------------------------------+-----------------------------------------------------------------------------+
| Session 1 | .. image:: _static/Connectopies_subject_100307_REST1_planes_X18-Y65-Z50.png | .. image:: _static/Connectopies_subject_100307_REST1_planes_X73-Y65-Z50.png |
+-----------+-----------------------------------------------------------------------------+-----------------------------------------------------------------------------+
| Session 2 | .. image:: _static/Connectopies_subject_100307_REST2_planes_X18-Y65-Z50.png | .. image:: _static/Connectopies_subject_100307_REST2_planes_X73-Y65-Z50.png |
+-----------+-----------------------------------------------------------------------------+-----------------------------------------------------------------------------+

**Connectopies of subject 103414 from three different planes orthogonal to the X, Y, and Z axes.**

+-----------+-----------------------------------------------------------------------------+-----------------------------------------------------------------------------+
|           |                             Left Hemisphire                                 |                             Right Hemisphire                                |
+-----------+-----------------------------------------------------------------------------+-----------------------------------------------------------------------------+
| Session 1 | .. image:: _static/Connectopies_subject_103414_REST1_planes_X25-Y55-Z69.png | .. image:: _static/Connectopies_subject_103414_REST1_planes_X66-Y55-Z69.png |
+-----------+-----------------------------------------------------------------------------+-----------------------------------------------------------------------------+
| Session 2 | .. image:: _static/Connectopies_subject_103414_REST2_planes_X25-Y55-Z69.png | .. image:: _static/Connectopies_subject_103414_REST2_planes_X66-Y55-Z69.png |
+-----------+-----------------------------------------------------------------------------+-----------------------------------------------------------------------------+

+-----------+-----------------------------------------------------------------------------+-----------------------------------------------------------------------------+
|           |                             Left Hemisphire                                 |                             Right Hemisphire                                |
+-----------+-----------------------------------------------------------------------------+-----------------------------------------------------------------------------+
| Session 1 | .. image:: _static/Connectopies_subject_103414_REST1_planes_X18-Y65-Z50.png | .. image:: _static/Connectopies_subject_103414_REST1_planes_X73-Y65-Z50.png |
+-----------+-----------------------------------------------------------------------------+-----------------------------------------------------------------------------+
| Session 2 | .. image:: _static/Connectopies_subject_103414_REST2_planes_X18-Y65-Z50.png | .. image:: _static/Connectopies_subject_103414_REST2_planes_X73-Y65-Z50.png |
+-----------+-----------------------------------------------------------------------------+-----------------------------------------------------------------------------+

**Connectopies of subject 105115 from three different planes orthogonal to the X, Y, and Z axes.**

+-----------+-----------------------------------------------------------------------------+-----------------------------------------------------------------------------+
|           |                             Left Hemisphire                                 |                             Right Hemisphire                                |
+-----------+-----------------------------------------------------------------------------+-----------------------------------------------------------------------------+
| Session 1 | .. image:: _static/Connectopies_subject_105115_REST1_planes_X25-Y55-Z69.png | .. image:: _static/Connectopies_subject_105115_REST1_planes_X66-Y55-Z69.png |
+-----------+-----------------------------------------------------------------------------+-----------------------------------------------------------------------------+
| Session 2 | .. image:: _static/Connectopies_subject_105115_REST2_planes_X25-Y55-Z69.png | .. image:: _static/Connectopies_subject_105115_REST2_planes_X66-Y55-Z69.png |
+-----------+-----------------------------------------------------------------------------+-----------------------------------------------------------------------------+

+-----------+-----------------------------------------------------------------------------+-----------------------------------------------------------------------------+
|           |                             Left Hemisphire                                 |                             Right Hemisphire                                |
+-----------+-----------------------------------------------------------------------------+-----------------------------------------------------------------------------+
| Session 1 | .. image:: _static/Connectopies_subject_105115_REST1_planes_X18-Y65-Z50.png | .. image:: _static/Connectopies_subject_105115_REST1_planes_X73-Y65-Z50.png |
+-----------+-----------------------------------------------------------------------------+-----------------------------------------------------------------------------+
| Session 2 | .. image:: _static/Connectopies_subject_105115_REST2_planes_X18-Y65-Z50.png | .. image:: _static/Connectopies_subject_105115_REST2_planes_X73-Y65-Z50.png |
+-----------+-----------------------------------------------------------------------------+-----------------------------------------------------------------------------+

Isomap embedding
~~~~~~~~~~~~~~~~

**Cross-session normalized root mean squared error for the left and right hemispheres.**

+---------+------------------------------------------------+
|         |                     nRMSE                      |
+ Subject +-----------------------+------------------------+
|         |    Left Hemisphere    |    Right Hemisphere    |
+=========+=======================+========================+
| 100307  | 0.01386               | 0.01731                |
+---------+-----------------------+------------------------+
| 103414  | 0.02312               | 0.01691                |
+---------+-----------------------+------------------------+
| 105115  | 0.01359               | 0.02198                |
+---------+-----------------------+------------------------+

**Cross-subject normalized root mean squared error for session 1 and session 2 and for the left and right hemispheres.**

+---------------+---------+------------------------------------------------+
|               |         |                     nRMSE                      |
| Subjects      | Session +-----------------------+------------------------+
|               |         |    Left Hemisphere    |    Right Hemisphere    |
+===============+=========+=======================+========================+
|               |       1 | 0.02317               | 0.01719                |
| 100307/103414 +---------+-----------------------+------------------------+
|               |       2 | 0.02763               | 0.02206                |
+---------------+---------+-----------------------+------------------------+
|               |       1 | 0.01925               | 0.02431                |
| 100307/105115 +---------+-----------------------+------------------------+
|               |       2 | 0.03177               | 0.03437                |
+---------------+---------+-----------------------+------------------------+
|               |       1 | 0.02374               | 0.02408                |
| 103414/105115 +---------+-----------------------+------------------------+
|               |       2 | 0.03310               | 0.03264                |
+---------------+---------+-----------------------+------------------------+


The following images show the connectopic mapping for subjects 100307, 103414 and 105115 from the `Human Connectome Project <https://www.humanconnectome.org>`_. The spatial distribution of the connectopies is very similar to Haak and colleagues' results.

**Connectopies of subject 100307 from three different planes orthogonal to the X, Y, and Z axes.**

+-----------+------------------------------------------------------------------------------------+------------------------------------------------------------------------------------+
|           |                             Left Hemisphire                                        |                             Right Hemisphire                                       |
+-----------+------------------------------------------------------------------------------------+------------------------------------------------------------------------------------+
| Session 1 | .. image:: _static/Connectopies_subject_100307_REST1_planes_X25-Y55-Z69_isomap.png | .. image:: _static/Connectopies_subject_100307_REST1_planes_X66-Y55-Z69_isomap.png |
+-----------+------------------------------------------------------------------------------------+------------------------------------------------------------------------------------+
| Session 2 | .. image:: _static/Connectopies_subject_100307_REST2_planes_X25-Y55-Z69_isomap.png | .. image:: _static/Connectopies_subject_100307_REST2_planes_X66-Y55-Z69_isomap.png |
+-----------+------------------------------------------------------------------------------------+------------------------------------------------------------------------------------+

+-----------+------------------------------------------------------------------------------------+------------------------------------------------------------------------------------+
|           |                             Left Hemisphire                                        |                             Right Hemisphire                                       |
+-----------+------------------------------------------------------------------------------------+------------------------------------------------------------------------------------+
| Session 1 | .. image:: _static/Connectopies_subject_100307_REST1_planes_X18-Y65-Z50_isomap.png | .. image:: _static/Connectopies_subject_100307_REST1_planes_X73-Y65-Z50_isomap.png |
+-----------+------------------------------------------------------------------------------------+------------------------------------------------------------------------------------+
| Session 2 | .. image:: _static/Connectopies_subject_100307_REST2_planes_X18-Y65-Z50_isomap.png | .. image:: _static/Connectopies_subject_100307_REST2_planes_X73-Y65-Z50_isomap.png |
+-----------+------------------------------------------------------------------------------------+------------------------------------------------------------------------------------+

**Connectopies of subject 103414 from three different planes orthogonal to the X, Y, and Z axes.**

+-----------+------------------------------------------------------------------------------------+------------------------------------------------------------------------------------+
|           |                             Left Hemisphire                                        |                             Right Hemisphire                                       |
+-----------+------------------------------------------------------------------------------------+------------------------------------------------------------------------------------+
| Session 1 | .. image:: _static/Connectopies_subject_103414_REST1_planes_X25-Y55-Z69_isomap.png | .. image:: _static/Connectopies_subject_103414_REST1_planes_X66-Y55-Z69_isomap.png |
+-----------+------------------------------------------------------------------------------------+------------------------------------------------------------------------------------+
| Session 2 | .. image:: _static/Connectopies_subject_103414_REST2_planes_X25-Y55-Z69_isomap.png | .. image:: _static/Connectopies_subject_103414_REST2_planes_X66-Y55-Z69_isomap.png |
+-----------+------------------------------------------------------------------------------------+------------------------------------------------------------------------------------+

+-----------+------------------------------------------------------------------------------------+------------------------------------------------------------------------------------+
|           |                             Left Hemisphire                                        |                             Right Hemisphire                                       |
+-----------+------------------------------------------------------------------------------------+------------------------------------------------------------------------------------+
| Session 1 | .. image:: _static/Connectopies_subject_103414_REST1_planes_X18-Y65-Z50_isomap.png | .. image:: _static/Connectopies_subject_103414_REST1_planes_X73-Y65-Z50_isomap.png |
+-----------+------------------------------------------------------------------------------------+------------------------------------------------------------------------------------+
| Session 2 | .. image:: _static/Connectopies_subject_103414_REST2_planes_X18-Y65-Z50_isomap.png | .. image:: _static/Connectopies_subject_103414_REST2_planes_X73-Y65-Z50_isomap.png |
+-----------+------------------------------------------------------------------------------------+------------------------------------------------------------------------------------+

**Connectopies of subject 105115 from three different planes orthogonal to the X, Y, and Z axes.**

+-----------+------------------------------------------------------------------------------------+------------------------------------------------------------------------------------+
|           |                             Left Hemisphire                                        |                             Right Hemisphire                                       |
+-----------+------------------------------------------------------------------------------------+------------------------------------------------------------------------------------+
| Session 1 | .. image:: _static/Connectopies_subject_105115_REST1_planes_X25-Y55-Z69_isomap.png | .. image:: _static/Connectopies_subject_105115_REST1_planes_X66-Y55-Z69_isomap.png |
+-----------+------------------------------------------------------------------------------------+------------------------------------------------------------------------------------+
| Session 2 | .. image:: _static/Connectopies_subject_105115_REST2_planes_X25-Y55-Z69_isomap.png | .. image:: _static/Connectopies_subject_105115_REST2_planes_X66-Y55-Z69_isomap.png |
+-----------+------------------------------------------------------------------------------------+------------------------------------------------------------------------------------+

+-----------+------------------------------------------------------------------------------------+------------------------------------------------------------------------------------+
|           |                             Left Hemisphire                                        |                             Right Hemisphire                                       |
+-----------+------------------------------------------------------------------------------------+------------------------------------------------------------------------------------+
| Session 1 | .. image:: _static/Connectopies_subject_105115_REST1_planes_X18-Y65-Z50_isomap.png | .. image:: _static/Connectopies_subject_105115_REST1_planes_X73-Y65-Z50_isomap.png |
+-----------+------------------------------------------------------------------------------------+------------------------------------------------------------------------------------+
| Session 2 | .. image:: _static/Connectopies_subject_105115_REST2_planes_X18-Y65-Z50_isomap.png | .. image:: _static/Connectopies_subject_105115_REST2_planes_X73-Y65-Z50_isomap.png |
+-----------+------------------------------------------------------------------------------------+------------------------------------------------------------------------------------+

.. [1] Haak et al., "Connectopic mapping with resting-state fMRI", 2016
.. [2] Van der Maaten, Hilton, "Visualizing High-Dimensional Data Using t-SNE", 2008


