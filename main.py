from arguments import generator_options
from arguments import LABEL_TO_CLASS
from generate_artificial_images import perform_augmentation
from visualizer import save_visuals
from saver import make_save_dirs
from get_backgrounds_and_data import fetch_image_gt_paths
from object_details import find_obj_loc_and_vals
from generate_artificial_images import get_locations_in_image
import cv2
import csv
import tqdm
from joblib import Parallel, delayed
import multiprocessing
import os
import numpy as np
from pascal_voc_writer import Writer


def read_files_and_visualize(data):
    """
    This function reads all the images and corresponding
    labels and calls the visualizer.
    :param data: List containing paths to images and labels
    :return: No returns.
    """

    image = cv2.imread(data[0])
    label = cv2.imread(data[1], 0)
    name = data[1].split('/')[-1].split('.')[0]
    obj_name = name[:-4]
    label_value = sorted(np.unique(label))[0]
    obj_details = find_obj_loc_and_vals(image, label,
                                        label_value, obj_name)
    obj_locations = get_locations_in_image(obj_details['obj_loc'])
    rect_points = [min(obj_locations[:, 1]), min(obj_locations[:, 0]),
                   max(obj_locations[:, 1]), max(obj_locations[:, 0])]
    obj_label = [[obj_name] + rect_points]
    save_visuals(image, label, obj_label, name)

    if generator_options.save_obj_det_label:
        img_path = data[0]
        img_dimension = generator_options.image_dimension
        writer = Writer(img_path, img_dimension[0],
                        img_dimension[1])
        [writer.addObject(*l) for l in obj_label]
        save_path = os.path.join(
            generator_options.obj_det_save_path,
            generator_options.name_format %
            (name) + '.xml')
        writer.save(save_path)


if __name__ == '__main__':

    if generator_options.mode == 1:
        perform_augmentation()
    else:
        make_save_dirs()
        data_paths = fetch_image_gt_paths()

        for data in tqdm.tqdm(data_paths,
                              desc='Saving visuals'):
            read_files_and_visualize(data)

        # num_cores = multiprocessing.cpu_count()
        # Parallel(n_jobs=num_cores)(delayed(read_files_and_visualize)
        #                            (data) for data in tqdm.tqdm(data_paths,
        #                                                         desc='Saving visuals'))
