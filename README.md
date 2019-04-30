## Match Corrrespondance of Vegetation Point Clouds

Implementation of a deep Siamese ConvNet to quantify/ predict the similarity of given three-dimensional points to find corresponding ones. The approach follows the work of '3DMatch' (link) and uses distances in a high-dimensional space as a measure of similarity i.e. correspondance. Truncated distance functions (TDFs) of the respective points are used as input to the network. The identified correspondances can then be used to reconstruct both the camera trajectory as well as the caputered object.

### Data

The data used for this work consisted of a large number of synthesized RGB-D images of vegetation.
Segmentations of the images have also been available for sampling and analyis purposes.
Due to signed NDAs I cannot make this data publicly available.

RGB-D images to be used with the code should be stored in .....

### Code

The code is written in Python and should be easily runnable by executing the main script.

### Results

Some of the resulting plots
