import glob
import numpy as np
import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
Image.MAX_IMAGE_PIXELS = None

def norm(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return 0
    return v/norm

def pad_im(cr_im, new_size, final_size=256, bkg_color='white'):
    padded_im = Image.new('RGB', (new_size, new_size), 'white')
    padded_im.paste(cr_im, ((new_size-cr_im.size[0])/2, (new_size-cr_im.size[1])/2))
    padded_im = padded_im.resize((final_size, final_size), Image.ANTIALIAS)
    return padded_im

def compute_normal_from_depth(im_arr):
    im_normal = np.zeros([im_arr.shape[0], im_arr.shape[1], 3])
    neighbors = [[-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1]]

    for i in range(1, im_arr.shape[0]-1):
        for j in range(1, im_arr.shape[1]-1):
            normal_acc = np.array([0.0, 0.0, 0.0])
            for k in range(8):
                n_b = neighbors[k]
                n_c = neighbors[(k+1)%8]

                p_a = np.array([i, j, im_arr[i, j]])
                p_b = np.array([i+n_b[0], j+n_b[1], im_arr[i+n_b[0], j+n_b[1]]])
                p_c = np.array([i+n_c[0], j+n_c[1], im_arr[i+n_c[0], j+n_c[1]]])
                
                normal =  -np.cross(p_b-p_a, p_c-p_a)
                if normal[2] < 0.0:
                    normal = -normal

                normal = norm(normal)
                normal_acc += normal

            normal_mean = normal_acc / 8.0
            normal_mean = norm(normal_mean)

            im_normal[i, j, 2] = round((normal_mean[0] + 1.0)/ 2.0 * 255.0)
            im_normal[i, j, 1] = round((normal_mean[1] + 1.0)/ 2.0 * 255.0) 
            im_normal[i, j, 0] = round(normal_mean[2] * 255.0 )

    return im_normal

final_size = 256
annot_paths = glob.glob('/media/nelson/Workspace1/Projects/building_reconstruction/2D_polygons_annotator/new_annots/*')
rgb_im = Image.open('/media/nelson/Workspace1/Projects/building_reconstruction/high_res_la/aligned_images/rgb_aligned.tif')
depth_im = Image.open('/media/nelson/Workspace1/Projects/building_reconstruction/high_res_la/aligned_images/norm_im_0.3m_aligned_raw.tif')
gray_im = Image.open('/media/nelson/Workspace1/Projects/building_reconstruction/high_res_la/aligned_images/gray_aligned.tif')
surf_im = Image.open('/media/nelson/Workspace1/Projects/building_reconstruction/high_res_la/aligned_images/surf_im_0.3m_aligned_raw.tif')

for a_path in annot_paths:
    with open(a_path) as f:

        # get im id
        im_id = a_path.split(' ')[-1].replace('.npy', '')
        print(im_id)

        # load annotation
        annot = np.load(f)
        annot = annot[()]

        # clean empty annotations
        graph = annot['graph']
        if len(graph.keys()) == 0:
            os.remove(a_path)
            continue

        # get coords
        pts = np.array(graph.keys())
        xs = [pt[0] for pt in graph.keys()]
        ys = [pt[1] for pt in graph.keys()]
        margin = 24

        lt = (np.min(xs)-margin, np.min(ys)-margin)
        rb = (np.max(xs)+margin, np.max(ys)+margin)

        # crop image
        curr_size = np.array([rb[0]-lt[0], rb[1]-lt[1]])
        new_size = int(np.max([np.max(curr_size), final_size]))
        x_c, y_c = (lt[0]+rb[0])/2, (lt[1]+rb[1])/2

        # distance from center
        dist = new_size/2

        # crop image
        cr_im = rgb_im.crop((lt[0], lt[1], rb[0], rb[1])).convert('RGB')
        cr_dp_im = depth_im.crop((lt[0], lt[1], rb[0], rb[1])).convert('L')
        cr_gray_im = gray_im.crop((lt[0], lt[1], rb[0], rb[1])).convert('L')
        cr_surf_im = surf_im.crop((lt[0], lt[1], rb[0], rb[1])).convert('RGB')
        # cr_im = rgb_im.crop((x_c-dist, y_c-dist, x_c+dist, y_c+dist)).resize((final_size, final_size), Image.ANTIALIAS).convert('RGB')
        # cr_dp_im = depth_im.crop((x_c-dist, y_c-dist, x_c+dist, y_c+dist)).resize((final_size, final_size), Image.ANTIALIAS).convert('L')
        # cr_gray_im = gray_im.crop((x_c-dist, y_c-dist, x_c+dist, y_c+dist)).resize((final_size, final_size), Image.ANTIALIAS).convert('L')
        # cr_surf_im = surf_im.crop((x_c-dist, y_c-dist, x_c+dist, y_c+dist)).resize((final_size, final_size), Image.ANTIALIAS).convert('RGB')

        cr_im = pad_im(cr_im, new_size, final_size=256, bkg_color='white')
        cr_dp_im = pad_im(cr_dp_im, new_size, final_size=256, bkg_color='black')
        cr_gray_im = pad_im(cr_gray_im, new_size, final_size=256, bkg_color='black')
        cr_surf_im = pad_im(cr_surf_im, new_size, final_size=256, bkg_color='white')

        # save images
        cr_im.save('/media/nelson/Workspace1/Projects/building_reconstruction/la_dataset/rgb/{}.jpg'.format(im_id))
        cr_dp_im.save('/media/nelson/Workspace1/Projects/building_reconstruction/la_dataset/depth/{}.jpg'.format(im_id))
        cr_gray_im.save('/media/nelson/Workspace1/Projects/building_reconstruction/la_dataset/gray/{}.jpg'.format(im_id))
        cr_surf_im.save('/media/nelson/Workspace1/Projects/building_reconstruction/la_dataset/surf/{}.jpg'.format(im_id))

        # transform annots
        shift_bb = np.array(lt)
        shift = np.array([(new_size-curr_size[0])/2, (new_size-curr_size[1])/2])
        scale = float(final_size)/new_size

        trans_graph = {}
        for k in graph.keys():
            new_k = tuple((k - shift_bb + shift)*scale)
            trans_graph[new_k] = []
            for pt in graph[k]:
                new_pt = (pt - shift_bb + shift)*scale
                trans_graph[new_k].append(new_pt) 


        np.save('/media/nelson/Workspace1/Projects/building_reconstruction/la_dataset/annots/{}.npy'.format(im_id), trans_graph, 'bytes')

        # compute gt outline
        im_out = Image.fromarray(np.zeros((256, 256))).convert('L')
        draw = ImageDraw.Draw(im_out)
        for pt1 in trans_graph.keys():
            for pt2 in trans_graph[pt1]:
                draw.line((pt1[0], pt1[1], pt2[0], pt2[1]), fill='white', width=2)
        im_out.save('/media/nelson/Workspace1/Projects/building_reconstruction/la_dataset/outline_gt/{}.jpg'.format(im_id))

        # debug
        # plt.figure()
        # cr_dp_im = np.array(cr_dp_im.convert('L'))*255.0
        # #sf_norm_im = compute_normal_from_depth(cr_dp_im)
        # #cr_dp_im = cr_dp_im*255.0/np.max(cr_dp_im)

        # # convert to PIL object
        # cr_dp_im = Image.fromarray(cr_dp_im.astype(np.uint8))
        # #sf_norm_im = Image.fromarray(sf_norm_im.astype(np.uint8))

        # draw = ImageDraw.Draw(cr_im)
        # for pt1 in trans_graph.keys():
        #     for pt2 in trans_graph[pt1]:
        #         draw.ellipse((pt1[0]-2, pt1[1]-2, pt1[0]+2, pt1[1]+2), fill='red')
        #         draw.ellipse((pt2[0]-2, pt2[1]-2, pt2[0]+2, pt2[1]+2), fill='red')    
        #         draw.line((pt1[0], pt1[1], pt2[0], pt2[1]), fill='blue', width=2)
        # plt.imshow(cr_im)
        # #plt.imshow(sf_norm_im)
        # plt.figure()
        # plt.imshow(cr_dp_im.convert('RGB'))
        # plt.show()