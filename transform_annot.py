import numpy as np
import glob

# assuming coordinates (widht, height)
annot_rgb_size = np.array((0.0, 7721.0, 0.0, 9282.0))
new_rgb_size = np.array((0.0, 14265.0, 0.0,14993.0))

# kml coords
annot_kml_coords = np.array((-118.322382, -118.276043, 34.075866, 34.037319))
new_kml_coords = np.array((-118.32238, -118.27604, 34.07787, 34.03732))

# compute deltas
annot_kml_delta = np.array((annot_kml_coords[1]-annot_kml_coords[0], annot_kml_coords[2]-annot_kml_coords[3]))
annot_rgb_delta = np.array((annot_rgb_size[1]-annot_rgb_size[0], annot_rgb_size[3]-annot_rgb_size[2]))
new_kml_delta = np.array((new_kml_coords[1]-new_kml_coords[0], new_kml_coords[2]-new_kml_coords[3]))
new_rgb_delta = np.array((new_rgb_size[1]-new_rgb_size[0], new_rgb_size[3]-new_rgb_size[2]))

# compute ratios
annot_kml_per_pixel = annot_kml_delta/annot_rgb_delta
new_kml_per_pixel = new_kml_delta/new_rgb_delta
kml_shift = np.abs(new_kml_coords - annot_kml_coords)

# compute scale and shift
annot_scale = annot_kml_per_pixel/new_kml_per_pixel
annot_shift = np.abs((kml_shift[0], kml_shift[2])/new_kml_per_pixel)+3

print(new_kml_per_pixel)
print(annot_kml_per_pixel)
print(kml_shift)



annots_paths = glob.glob('./annots/*')

for annot_p in annots_paths:
    with open(annot_p) as f:
        annot = np.load(f)

        # define shift
        shift = annot_shift
        scale = annot_scale
        annot = annot[()]

        # fix previous corner]
        if annot['prevCorner'] is not None:
            print(annot['prevCorner'])
            annot['prevCorner'] = tuple(np.array(annot['prevCorner'])*scale + shift)

        # fix neighbours
        new_graph = {}
        for p in annot['graph'].keys():
            new_neighbours = []
            for n in annot['graph'][p]:
                new_neighbours.append(tuple(np.array(n)*scale+shift))
            new_graph[tuple(np.array(p)*scale+shift)] = new_neighbours
        annot['graph'] = new_graph

        # fix edges tracker
        for k in range(len(annot['edgesTracker'])):
            p1, p2 = annot['edgesTracker'][k]
            annot['edgesTracker'][k] = (tuple(np.array(p1)*scale-shift), tuple(np.array(p2)*scale+shift))

        # fix selected corner
        if annot['selectedCorner'] is not None:
            annot['selectedCorner'] = tuple(np.array(annot['selectedCorner'])*scale+shift)

    dst_annot_p = annot_p.replace('annots', 'fixed_annots')
    np.save(dst_annot_p, annot)
    #print(dst_annot_p)
