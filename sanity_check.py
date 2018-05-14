import glob
import numpy as np

annot_paths = glob.glob('/media/nelson/Workspace1/Projects/building_reconstruction/2D_polygons_annotator/fixed_annots/*')

for a_path in annot_paths:
    with open(a_path) as f:

        # load annotation
        annot = np.load(f)
        annot = annot[()]

        graph = annot['graph']