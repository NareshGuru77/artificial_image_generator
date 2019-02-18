from arguments import generator_options
from visualizer import save_visuals
import os
import cv2
import csv
from pascal_voc_writer import Writer


def make_save_dirs():
    """
    This function checks whether the save paths exists. Creates them if
    they do not exist.
    :return: No returns.
    """

    if generator_options.mode == 1:
        if not os.path.isdir(generator_options.image_save_path):
            os.makedirs(generator_options.image_save_path)

        if not os.path.isdir(generator_options.label_save_path):
            os.makedirs(generator_options.label_save_path)

    if generator_options.save_obj_det_label:
        if not os.path.isdir(generator_options.obj_det_save_path):
            os.makedirs(generator_options.obj_det_save_path)

    if generator_options.save_mask:
        if not os.path.isdir(generator_options.mask_save_path):
            os.makedirs(generator_options.mask_save_path)

    if generator_options.save_label_preview:
        if not os.path.isdir(generator_options.preview_save_path):
            os.makedirs(generator_options.preview_save_path)

    if generator_options.save_overlay:
        if not os.path.isdir(generator_options.overlay_save_path):
            os.makedirs(generator_options.overlay_save_path)


def save_data(artificial_image, semantic_label, obj_det_label, index):
    """
    This function saves the artificial image and its corresponding semantic
    label. Also saves object detection labels, plot preview and segmentation
    mask images based on "generator_options".

    :param artificial_image: The artificial image which needs to be saved.
    :param semantic_label: The semantic segmentation label image which
                           needs to be saved.
    :param obj_det_label: The object detection label which needs to be
                          saved. Can be None if "save_obj_det_label" is false.
    :param index: The index value to be included in the name of the files.
    :return: No returns.
    """

    cv2.imwrite(os.path.join(
        generator_options.image_save_path,
        generator_options.name_format %
        (index + generator_options.start_index) + '.jpg'),
        artificial_image)

    cv2.imwrite(os.path.join(
        generator_options.label_save_path,
        generator_options.name_format %
        (index + generator_options.start_index) + '.png'),
        semantic_label)

    if generator_options.save_obj_det_label:
        img_path = os.path.join(
            generator_options.image_save_path,
            generator_options.name_format %
            (index + generator_options.start_index) + '.jpg')
        img_dimension = generator_options.image_dimension
        writer = Writer(img_path, img_dimension[0],
                        img_dimension[1])
        [writer.addObject(*l) for l in obj_det_label]
        save_path = os.path.join(
                generator_options.obj_det_save_path,
                generator_options.name_format %
                (index + generator_options.start_index) + '.xml')
        writer.save(save_path)
        # with open(os.path.join(
        #         generator_options.obj_det_save_path,
        #         generator_options.name_format %
        #         (index + generator_options.start_index) + '.csv'), 'w') as f:
        #
        #     wr = csv.writer(f, delimiter=',')
        #     [wr.writerow(l) for l in obj_det_label]
    else:
        obj_det_label = None

    if (generator_options.save_mask or
            generator_options.save_label_preview or
            generator_options.save_overlay):
        save_visuals(artificial_image, semantic_label,
                     obj_det_label, index)
