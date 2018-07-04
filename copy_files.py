import os
from shutil import copyfile

with open('/media/nelson/Workspace1/Projects/building_reconstruction/la_dataset/valid_list.txt') as f:
    for line in f.readlines():
        src = os.path.join('/media/nelson/Workspace1/Projects/building_reconstruction/2D_polygons_annotator/fixed_annots/', 'annot- ' + line.strip() + '.npy')
        dst = os.path.join('/media/nelson/Workspace1/Projects/building_reconstruction/2D_polygons_annotator/annots_valid/', 'annot- ' + line.strip() + '.npy')
        copyfile(src, dst)