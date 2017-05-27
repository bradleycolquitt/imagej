"""
Script to measure object properties (intensity, area) and intersection with user defined ROI (ImageJ)
Gaussian filter then Triangle thresholding prodcued best object isolation, by trial and error
"""

import sys, os, re, copy
import ijroi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage as ndi
from skimage.filters import sobel, gaussian
from skimage import measure
from skimage.measure import label
from skimage.color import label2rgb
from skimage.feature import canny
from shapely.geometry import box
from shapely.geometry import Polygon
from skimage.filters import threshold_triangle

from itertools import compress

def get_resolution(image_fname):
    im1 = skimage.external.tifffile.TiffFile(image_fname)
    info = im1.info()
    im1.close()
    sel = None
    for line in a.splitlines():
        if re.search("voxel_size_x", line):
            sel = float(line.split(":")[1].strip())
    return(sel)

def threshold_image_group(images, channel, min_value = 0, filter_gaussian=False):
    threshes = []
    ims_slice = [im[0,0,channel,:,:] for im in images]

    if filter_gaussian:
        ims_slice = [gaussian_filter(im) for im in ims_slice]

    im_concat = np.concatenate(ims_slice)
    im_concat[im_concat <= min_value] = 0
    #thresh = threshold_otsu(im_concat)
    thresh = threshold_triangle(im_concat)
    print(thresh)
    threshes.append(thresh)

    ims_thresh = []
    for j in range(len(images)):
        ims_thresh.append(ims_slice[j] > thresh)
    return(ims_thresh)

def gaussian_filter(image, sigma=1):
    image_filtered = gaussian(image, sigma=sigma)
    return(image_filtered)

def find_edges(image):
    edges = canny(image/255.)
    fill_image = ndi.binary_fill_holes(edges)
    return edges, fill_image

def label_image(image, min_size=20):
    labels, num_features = ndi.label(image)
    print("Number of features %s" % num_features)
    return labels, num_features

def extract_properties(labels, image):
    """ Get intensity, area, and bounding box info from labels """
    properties = measure.regionprops(labels, image)
    intens = [p.mean_intensity for p in properties]
    areas = [p.area for p in properties]
    bboxes = [box(p.bbox[0], p.bbox[1], p.bbox[2], p.bbox[3]) for p in properties]
    return(intens, areas, bboxes)

def plot_segmented_images(image, edges, fill_image, labels, out_prefix):
    """ Write out thresholded images for evaluation """
    fig, axes1 = plt.subplots(ncols=2, nrows=2, figsize=(24,24))
    axes = axes1.ravel()
    axes[0].imshow(image)
    axes[1].imshow(edges)
    axes[2].imshow(fill_image)
    image_label_overlay = label2rgb(labels, image=image)
    axes[3].imshow(image_label_overlay)
    #plt.show()
    plt.savefig("%s_threshold.jpg" % out_prefix)
    plt.close()

def read_user_roi(roi_fname):
    """ Import ImageJ defined ROI and convert it to Shapely polygon """
    if not os.path.exists(roi_fname):
        return None
    roi_obj = open(roi_fname, 'rb')
    roi = ijroi.read_roi(roi_obj)
    roi_poly = Polygon(roi)
    return(roi_poly)

def intersect_regions(roi, bboxes):
    """ Intersect objects with user-defined ROI """
    bboxes_inter = [roi.intersection(b).area for b in bboxes]
    bboxes_inter_bool = [b > 0 for b in bboxes_inter] 
    return(bboxes_inter_bool)

def process_images(image, image_threshed, image_fname, channel, out_dir):
    image_prefix = image_fname.strip(".lsm")
    image_basename = os.path.basename(image_prefix)

    # Read in manual ROI
    roi_poly = read_user_roi("%s.roi" % image_prefix)
    if roi_poly is None:
        return None

    # Find edges, labels
    print("Labeling image")
    #edges, fill_image = find_edges(image)
    labels, num_features = label_image(image_threshed)

    # Output thresholded images
    print("Writing thresholded images")
    threshold_fname = "/".join([out_dir, "_".join([image_basename, str(channel)])])
    plot_segmented_images(image_threshed, edges, fill_image, labels, threshold_fname)

    # Calculate properties
    intensities, areas, bboxes = extract_properties(labels, image)

    # Filter out small labels
    min_size = 20
    areas_mask = np.array(areas) > min_size
    #print(areas_mask)
    intensities = np.array(intensities)[areas_mask]
    areas = np.array(areas)[areas_mask]
    bboxes = list(compress(bboxes, areas_mask))
    print("Number of features of size %s: %s" % (min_size, len(intensities)))


    # intersect labels with manual ROI
    intersections = intersect_regions(roi_poly, bboxes)

    # Areas
    res = np.power(get_resolution(image_fname) * 1E6, 2) ## convert to um^2
    roi_area = roi_poly.area * res
    image_area = np.product(np.shape(image)) * res

    # add data to group DataFrame
    df = pd.DataFrame({
    'image' : image_basename,
    'id' : [i for i in range(len(intensities))],
    'mean' : intensities,
    'area' : areas,
    'in_roi' : intersections,
    'roi_area' : roi_area,
    'image_area' : image_area})

    return(df)

def main(argv):
    # Read in images
    data_dir = '/media/data/ish/170511_deaf_crhbp_sst_pv_hcr'
    patterns = ['lsm', 'ra|hvc']
    image_fnames = os.listdir(data_dir)
    for p in patterns:
        image_fnames = [im for im in image_fnames if re.search(p, im)]
    image_fnames = ["/".join([data_dir, im]) for im in image_fnames]
    print("Number of images to process: %s" % len(image_fnames))

    images = [skimage.external.tifffile.imread(im) for im in image_fnames]

    out_df = None
    out_dir = "/".join([data_dir, "analysis_threshed"])

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
    # Loop through channels
    nchannels = np.shape(images[0])[2]
    #nchannels = [1,2]
    for channel in range(nchannels):
        #for channel in nchannels:
        print("Channel %s" % channel)

        print("Thresholding image group...")
        images_threshed = threshold_image_group(images, channel, min_value=0, filter_gaussian=True)

        # Loop through channels
        for i in range(len(images_threshed)):
            image_cur = images[i][0,0,channel,:,:]
            image_threshed_cur = images_threshed[i]
            image_fname = image_fnames[i]
            print("Processing image: %s" % os.path.basename(image_fname))
            df = process_images(image_cur, image_threshed_cur, image_fname, channel, out_dir)

            if df is None:
                continue
            df['channel'] = channel
            if out_df is None:
                out_df = df
            else:
                out_df = out_df.append(df)


            print("Writing results.")
            out_fname = "/".join([out_dir, "results.csv"])
            out_df.to_csv(out_fname)


if __name__ == "__main__":
    main(sys.argv)
