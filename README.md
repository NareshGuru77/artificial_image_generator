# Usage:

## Mode 1:
python3 main.py ./images/ ./labels/ --backgrounds_path ./backgrounds/ --image_save_path ./augmented/images/ --label_save_path ./augmented/labels/ --save_label_preview True --preview_save_path ./augmented/preview/ --save_overlay True --overlay_save_path ./augmented/overlay/ --save_obj_det_label True --obj_det_save_path ./augmented/obj_det_labels/ --save_mask True --mask_save_path ./augmented/mask/ --num_images 8

## Mode 2:
python3 main.py ./images/ ./labels/ --backgrounds_path ./backgrounds/ --save_obj_det_label True --obj_det_save_path ./obj_det_labels --mode 2 --name_format %s

# Parameters:
description of arguments could be found in arguments.py
