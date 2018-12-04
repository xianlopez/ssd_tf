import tensorflow as tf
import os
import cv2
import tools
import numpy as np
import sys


####### DATA AUGMENTATION ########
class DataAugOpts:
    apply_data_augmentation = False  # If false, none of the following options have any effect.
    horizontal_flip = False
    vertical_flip = False
    random_brightness = False
    brightness_prob = 0.5
    brightness_delta_lower = -32
    brightness_delta_upper = 32
    random_contrast = False
    contrast_prob = 0.5
    contrast_factor_lower = 0.5
    contrast_factor_upper = 1.5
    random_saturation = False
    saturation_prob = 0.5
    saturation_factor_lower = 0.5
    saturation_factor_upper = 1.5
    random_hue = False
    hue_prob = 0.5
    hue_delta_lower = -0.1
    hue_delta_upper = 0.1
    rtss_rc = False  # Resize To Smallest Side and Random Crop
    smallest_side = 256
    random_crop = False  # This options and rtss_rc cannot be chosen at the same time.
    random_crop_proportion = 0.7
    random_crop_prob = 0.2
    sample_patch_like_ssd = False
    sample_xian = False
    sample_ssd_keras = False
    rotation_degree = 0
    rotation_prob = 0.5
    fully_rotate_bugs = False
    convert_to_grayscale_prob = 0
    ssd_original_pipeline = False
##################################


class DetectionDataAugmentation:
    def __init__(self, args, input_width, input_height, nclasses):
        self.input_width = input_width
        self.input_height = input_height
        self.data_aug_opts = args.data_aug_opts
        self.nclasses = nclasses
        self.outdir = args.outdir
        self.write_image_after_data_augmentation = args.write_image_after_data_augmentation
        if self.data_aug_opts.rotation_degree != 0:
            raise Exception('Rotation data agumentation can be used only in classification mode.')
        if self.data_aug_opts.fully_rotate_bugs:
            raise Exception('Rotation data agumentation can be used only in classification mode.')
        if args.num_workers > 1 and args.write_image_after_data_augmentation:
            raise Exception('Option write_image_after_data_augmentation is not compatible with more than one worker to load data.')
        if self.data_aug_opts.sample_patch_like_ssd and self.data_aug_opts.sample_xian:
            raise Exception('Options sample_patch_like_ssd and sample_xian are not compatible.')


    def data_augmenter(self, image, bboxes, filename):
        # Full pipelines:
        if self.data_aug_opts.ssd_original_pipeline:
            image, bboxes = self.ssd_original_pipeline(image, bboxes)
        else:
            # Cropping and stuff:
            if self.data_aug_opts.sample_xian:
                (image, bboxes) = tf.py_func(sample_xian_func, [image, bboxes], (tf.float32, tf.float32))
                image.set_shape((None, None, 3))
                bboxes.set_shape((None, 5))
            if self.data_aug_opts.sample_ssd_keras:
                (image, bboxes) = tf.py_func(expand_and_crop_ssd, [image, bboxes], (tf.float32, tf.float32))
                image.set_shape((None, None, 3))
                bboxes.set_shape((None, 5))
            if self.data_aug_opts.sample_patch_like_ssd:
                (image, bboxes) = tf.py_func(sample_like_ssd, [image, bboxes], (tf.float32, tf.float32))
                image.set_shape((None, None, 3))
                bboxes.set_shape((None, 5))
            # Photometric distortions:
            if self.data_aug_opts.random_brightness:
                image = random_adjust_brightness(image, self.data_aug_opts.brightness_delta_lower,
                                                 self.data_aug_opts.brightness_delta_upper,
                                                 self.data_aug_opts.brightness_prob)
            if self.data_aug_opts.random_contrast:
                image = random_adjust_contrast(image, self.data_aug_opts.contrast_factor_lower,
                                               self.data_aug_opts.contrast_factor_upper,
                                               self.data_aug_opts.contrast_prob)
            if self.data_aug_opts.random_saturation:
                image = random_adjust_saturation(image, self.data_aug_opts.saturation_factor_lower,
                                                 self.data_aug_opts.saturation_factor_upper,
                                                 self.data_aug_opts.saturation_prob)
            if self.data_aug_opts.random_hue:
                image = random_adjust_hue(image, self.data_aug_opts.hue_delta_lower,
                                          self.data_aug_opts.hue_delta_upper,
                                          self.data_aug_opts.hue_prob)
            if self.data_aug_opts.convert_to_grayscale_prob > 0:
                image = convert_to_grayscale(image, self.data_aug_opts.convert_to_grayscale_prob)
            # Flips:
            if self.data_aug_opts.horizontal_flip:
                flag = tf.random_uniform(()) < 0.5
                bboxes = tf.cond(flag, lambda: self.flip_boxes_horizontally(bboxes), lambda: tf.identity(bboxes))
                image = tf.cond(flag, lambda: tf.image.flip_left_right(image), lambda: tf.identity(image))
            if self.data_aug_opts.vertical_flip:
                flag = tf.random_uniform(()) < 0.5
                bboxes = tf.cond(flag, lambda: self.flip_boxes_vertically(bboxes), lambda: tf.identity(bboxes))
                image = tf.cond(flag, lambda: tf.image.flip_up_down(image), lambda: tf.identity(image))
        # Write images (for verification):
        if self.write_image_after_data_augmentation:
            image = tf.py_func(self.write_image, [image, bboxes, filename], tf.float32)
            image.set_shape((None, None, 3))
        return image, bboxes, filename

    def ssd_photometrics_sequence1(self, image):
        image = random_adjust_brightness(image, self.data_aug_opts.brightness_delta_lower, self.data_aug_opts.brightness_delta_upper, self.data_aug_opts.brightness_prob)
        image = random_adjust_contrast(image, self.data_aug_opts.contrast_factor_lower, self.data_aug_opts.contrast_factor_upper, self.data_aug_opts.contrast_prob)
        image = random_adjust_saturation(image, self.data_aug_opts.saturation_factor_lower, self.data_aug_opts.saturation_factor_upper, self.data_aug_opts.saturation_prob)
        image = random_adjust_hue(image, self.data_aug_opts.hue_delta_lower, self.data_aug_opts.hue_delta_upper, self.data_aug_opts.hue_prob)
        return image

    def ssd_photometrics_sequence2(self, image):
        image = random_adjust_brightness(image, self.data_aug_opts.brightness_delta_lower, self.data_aug_opts.brightness_delta_upper, self.data_aug_opts.brightness_prob)
        image = random_adjust_saturation(image, self.data_aug_opts.saturation_factor_lower, self.data_aug_opts.saturation_factor_upper, self.data_aug_opts.saturation_prob)
        image = random_adjust_hue(image, self.data_aug_opts.hue_delta_lower, self.data_aug_opts.hue_delta_upper, self.data_aug_opts.hue_prob)
        image = random_adjust_contrast(image, self.data_aug_opts.contrast_factor_lower, self.data_aug_opts.contrast_factor_upper, self.data_aug_opts.contrast_prob)
        return image

    def ssd_photometric_distortions(self, image):
        image = tf.cond(tf.greater(tf.random_uniform(shape=()), 0.5), lambda: self.ssd_photometrics_sequence1(image), lambda: self.ssd_photometrics_sequence2(image))
        return image

    def ssd_original_pipeline(self, image, bboxes):
        # Photometric distortions:
        image = self.ssd_photometric_distortions(image)
        # Expand and crop:
        (image, bboxes) = tf.py_func(expand_and_crop_ssd, [image, bboxes], (tf.float32, tf.float32))
        image.set_shape((None, None, 3))
        bboxes.set_shape((None, 5))
        # Random flip:
        flag = tf.random_uniform(()) < 0.5
        bboxes = tf.cond(flag, lambda: self.flip_boxes_horizontally(bboxes), lambda: tf.identity(bboxes))
        image = tf.cond(flag, lambda: tf.image.flip_left_right(image), lambda: tf.identity(image))
        return image, bboxes

    def flip_boxes_vertically(self, bboxes):
        # bboxes: (nboxes, 5)
        new_y_min = 1.0 - bboxes[:, 2] - bboxes[:, 4]  # (nboxes)
        before = bboxes[:, :2]  # (nboxes, 2)
        after = bboxes[:, 3:]  # (nboxes, 2)
        bboxes = tf.concat([before, tf.expand_dims(new_y_min, axis=1), after], axis=1)  # (nboxes, 5)
        return bboxes

    def flip_boxes_horizontally(self, bboxes):
        # bboxes: (nboxes, 5)
        new_x_min = 1.0 - bboxes[:, 1] - bboxes[:, 3]  # (nboxes)
        before = tf.expand_dims(bboxes[:, 0], axis=1)  # (nboxes, 2)
        after = bboxes[:, 2:]  # (nboxes, 2)
        bboxes = tf.concat([before, tf.expand_dims(new_x_min, axis=1), after], axis=1)  # (nboxes)
        return bboxes

    def write_image(self, image, bboxes, filename):
        filename_str = filename.decode(sys.getdefaultencoding())
        file_path_candidate = os.path.join(self.outdir, 'image_after_data_aug_' + filename_str + '.png')
        file_path = tools.ensure_new_path(file_path_candidate)
        print('path to save image: ' + file_path)
        print(str(np.min(image)) + '   ' + str(np.mean(image)) + '   ' + str(np.max(image)))
        img = image.copy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = tools.add_bounding_boxes_to_image(img, bboxes)
        cv2.imwrite(file_path, img)
        return image


def sample_xian_func(image, bboxes):
    # image: (image_height, image_width, 3)
    # bboxes: (nboxes, 5), [class_id, x_min, y_min, width, height]
    # print('sample_xian_func. image dtype: ' + str(image.dtype))
    min_scale = 0.1
    max_scale = 1
    min_aspect_ratio = 0.5
    max_aspect_ratio = 2
    rnd1 = np.random.randint(3)
    if rnd1 == 0:  # Use the entire original input image.
        pass
    elif rnd1 == 1:  # Sample a patch focusing in one object.
        # We will choose randomly the size and location of the patch, but making sure it contains the selected object.
        nboxes = bboxes.shape[0]
        if nboxes > 0:
            rnd2 = np.random.randint(nboxes)
            image, bboxes = sample_focusing_on_object(image, bboxes, rnd2)
        else:
            image, bboxes = sample_random_patch(image, bboxes, min_scale, max_scale, min_aspect_ratio, max_aspect_ratio)
    else:  # Randomly sample a patch.
        image, bboxes = sample_random_patch(image, bboxes, min_scale, max_scale, min_aspect_ratio, max_aspect_ratio)
    # print('sample_xian_func listo')
    return image, bboxes


def compute_coords_focusing_on_object(focus_box):
    min_scale_factor = 1.5
    max_scale_factor = 4
    max_side = max(focus_box[3], focus_box[4])
    patch_side = np.random.rand() * max_side * (max_scale_factor - min_scale_factor) + max_side * min_scale_factor
    x0_min = max(0, focus_box[1] + focus_box[3] - patch_side)
    x0_max = focus_box[1]
    patch_x0_rel = np.random.rand() * (x0_max - x0_min) + x0_min
    y0_min = max(0, focus_box[2] + focus_box[4] - patch_side)
    y0_max = focus_box[2]
    patch_y0_rel = np.random.rand() * (y0_max - y0_min) + y0_min
    # Crop the patch size if it goes outside the image, still in relative coordinates:
    patch_width_rel = min(patch_side, 1 - patch_x0_rel)
    patch_height_rel = min(patch_side, 1 - patch_y0_rel)
    return patch_x0_rel, patch_y0_rel, patch_width_rel, patch_height_rel


def sample_focusing_on_object(image, bboxes, box_idx):
    # image: (image_height, image_width, 3)
    # bboxes: (nboxes, 5), [class_id, x_min, y_min, width, height]
    iou_lower_bound = 0.2
    iou_upper_bound = 0.6
    remaining_attemps = 10
    focus_box = bboxes[box_idx]
    # We will try not to intersect another object in the middle more or less. We prefer to have clearly inside, or
    # not to have it at all almost.
    iterate = True
    while iterate and remaining_attemps > 0:
        remaining_attemps -= 1
        patch_x0_rel, patch_y0_rel, patch_width_rel, patch_height_rel = compute_coords_focusing_on_object(focus_box)
        fraction_contained_vec = compute_fraction_contained_with_all_boxes(bboxes, patch_x0_rel, patch_y0_rel, patch_width_rel, patch_height_rel)
        iterate = not np.logical_or(fraction_contained_vec < iou_lower_bound, iou_upper_bound < fraction_contained_vec).all()
    # Crop:
    img_height, img_width, _ = image.shape
    image, bboxes = sample_patch(image, bboxes, img_width, img_height, patch_x0_rel, patch_y0_rel, patch_width_rel, patch_height_rel)
    return image, bboxes


def expand_and_crop_ssd(image, bboxes):
    # image: (image_height, image_width, 3)
    # bboxes: (nboxes, 5), [class_id, x_min, y_min, width, height]

    img_height, img_width, _ = image.shape

    ####################
    ### Expand:
    rnd1 = np.random.randint(2)
    if rnd1 == 0:
        scale = np.random.rand() * 3.0 + 1 # between 1 and 4
        new_width = np.round(img_width * scale).astype(np.int32)
        new_height = np.round(img_height * scale).astype(np.int32)
        canvas = np.zeros(shape=(new_height, new_width, 3), dtype=np.float32)
        canvas[:, :, 0] = np.mean(image[:, :, 0])
        canvas[:, :, 1] = np.mean(image[:, :, 1])
        canvas[:, :, 2] = np.mean(image[:, :, 2])
        pos_i = np.random.randint(new_height - img_height + 1)
        pos_j = np.random.randint(new_width - img_width + 1)
        canvas[pos_i:(pos_i+img_height), pos_j:(pos_j+img_width), :] = image
        image = canvas
        bboxes[:, 1] = (pos_j + bboxes[:, 1] * img_width) / float(new_width)
        bboxes[:, 2] = (pos_i + bboxes[:, 2] * img_height) / float(new_height)
        bboxes[:, 3] = bboxes[:, 3] / scale
        bboxes[:, 4] = bboxes[:, 4] / scale
    else:
        new_width = img_width
        new_height = img_height

    ####################
    ### Random crop:
    min_scale = 0.3
    max_scale = 1
    min_aspect_ratio = 0.5
    max_aspect_ratio = 2
    iou_th_list = [-1, 0.1, 0.3, 0.5, 0.7, 0.9]
    while True: # Keep going until we either find a valid patch or return the original image.
        if np.random.rand() >= (1 - 0.857):
            iou_th = iou_th_list[np.random.randint(len(iou_th_list))]
            for _ in range(50):
                patch_x0_rel, patch_y0_rel, patch_width_rel, patch_height_rel = make_patch_shape(new_width, new_height, min_scale, max_scale, min_aspect_ratio, max_aspect_ratio)
                patch_x1_rel = patch_x0_rel + patch_width_rel
                patch_y1_rel = patch_y0_rel + patch_height_rel
                # Check boxes' IOU:
                patch_is_valid = False
                for i in range(bboxes.shape[0]):
                    x_center = bboxes[i, 1] + float(bboxes[i, 3]) / 2
                    y_center = bboxes[i, 2] + float(bboxes[i, 4]) / 2
                    if patch_x0_rel < x_center < patch_x1_rel and patch_y0_rel < y_center < patch_y1_rel:
                        if tools.compute_iou([patch_x0_rel, patch_y0_rel, patch_width_rel, patch_height_rel], bboxes[i, 1:]) > iou_th:
                            patch_is_valid = True
                            break
                if patch_is_valid:
                    image, bboxes = sample_patch(image, bboxes, new_width, new_height, patch_x0_rel, patch_y0_rel, patch_width_rel, patch_height_rel)
                    return image, bboxes

        else:
            return image, bboxes


def sample_like_ssd(image, bboxes):
    # image: (image_height, image_width, 3)
    # bboxes: (nboxes, 5), [class_id, x_min, y_min, width, height]
    min_scale = 0.1
    max_scale = 1
    min_aspect_ratio = 0.5
    max_aspect_ratio = 2
    iou_limit_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    max_sample_attemps = 100
    rnd1 = np.random.randint(3)
    if rnd1 == 0:  # Use the entire original input image.
        pass
    elif rnd1 == 1:  # Sample a patch so that the minimum IoU with the objects is 0.1, 0.3, 0.5, 0.7 or 0.9.
        # Maybe this implementation is not exactly what they do in the SSD paper, but should be similar...
        if bboxes.shape[0] > 0:
            img_height, img_width, _ = image.shape
            min_iou = 0
            iou_limit_idx = np.random.randint(len(iou_limit_list))
            iou_limit = iou_limit_list[iou_limit_idx]
            remaining_attemps = max_sample_attemps
            remaining_iou_changes = len(iou_limit_list) - 1
            while min_iou < iou_limit:
                remaining_attemps -= 1
                if remaining_attemps < 0:
                    iou_limit_idx = (iou_limit_idx + 1) % len(iou_limit_list)
                    iou_limit = iou_limit_list[iou_limit_idx]
                    remaining_attemps = max_sample_attemps
                    remaining_iou_changes -= 1
                    if remaining_iou_changes < 0:
                        # If too many attemps, desist and keep the current sample:
                        # print('desisting')
                        break
                patch_x0_rel, patch_y0_rel, patch_width_rel, patch_height_rel = make_patch_shape(img_width, img_height, min_scale, max_scale, min_aspect_ratio, max_aspect_ratio)
                min_iou = compute_minimum_iou_of_boxes_inside(bboxes, patch_x0_rel, patch_y0_rel, patch_width_rel, patch_height_rel)
            # print(min_iou)
            image, bboxes = sample_patch(image, bboxes, img_width, img_height, patch_x0_rel, patch_y0_rel, patch_width_rel, patch_height_rel)
        else:
            pass
    else:  # Randomly sample a patch.
        image, bboxes = sample_random_patch(image, bboxes, min_scale, max_scale, min_aspect_ratio, max_aspect_ratio)
    return image, bboxes


def make_patch_shape(img_width, img_height, min_scale, max_scale, min_aspect_ratio, max_aspect_ratio):
    scale = np.random.rand() * (max_scale - min_scale) + min_scale
    aspect_ratio = np.random.rand() * (max_aspect_ratio - min_aspect_ratio) + min_aspect_ratio
    patch_width = np.sqrt(aspect_ratio * scale * float(img_width) * float(img_height))
    patch_height = np.sqrt(scale * float(img_width) * float(img_height) / aspect_ratio)
    patch_width = np.minimum(np.maximum(np.round(patch_width), 1), img_width).astype(np.int32)
    patch_height = np.minimum(np.maximum(np.round(patch_height), 1), img_height).astype(np.int32)
    x0 = np.random.randint(img_width - patch_width + 1)
    y0 = np.random.randint(img_height - patch_height + 1)
    # Convert to relative coordinates:
    x0 = x0 / float(img_width)
    y0 = y0 / float(img_height)
    patch_width = patch_width / float(img_width)
    patch_height = patch_height / float(img_height)
    return x0, y0, patch_width, patch_height


def compute_fraction_contained_with_all_boxes(bboxes, patch_x0_rel, patch_y0_rel, patch_width_rel, patch_height_rel):
    nboxes = bboxes.shape[0]
    fraction_contained_vec = np.zeros(shape=nboxes, dtype=np.float32)
    for i in range(nboxes):
        fraction_contained_vec[i] = tools.compute_fraction_contained(bboxes[i, 1:], [patch_x0_rel, patch_y0_rel, patch_width_rel, patch_height_rel])
    return fraction_contained_vec


# Here, to compute the minimum IoU, we only take into account the boxes that would be included inside the crop.
# If no boxes are included, we return 0.
def compute_minimum_iou_of_boxes_inside(bboxes, patch_x0_rel, patch_y0_rel, patch_width_rel, patch_height_rel):
    nboxes = bboxes.shape[0]
    iou_list = []
    patch_x1_rel = patch_x0_rel + patch_width_rel
    patch_y1_rel = patch_y0_rel + patch_height_rel
    for i in range(nboxes):
        x_center = bboxes[i, 1] + float(bboxes[i, 3]) / 2
        y_center = bboxes[i, 2] + float(bboxes[i, 4]) / 2
        if patch_x0_rel < x_center < patch_x1_rel and patch_y0_rel < y_center < patch_y1_rel:
            iou_list.append(tools.compute_iou([patch_x0_rel, patch_y0_rel, patch_width_rel, patch_height_rel], bboxes[i, 1:]))
    if len(iou_list) > 0:
        min_iou = np.min(np.array(iou_list))
    else:
        min_iou = 0
    return min_iou


def sample_patch(image, bboxes, img_width, img_height, patch_x0_rel, patch_y0_rel, patch_width_rel, patch_height_rel):
    # Convert to absolute coordinates:
    patch_x0_abs = max(np.round(patch_x0_rel * img_width).astype(np.int32), 0)
    patch_y0_abs = max(np.round(patch_y0_rel * img_height).astype(np.int32), 0)
    patch_width_abs = min(np.round(patch_width_rel * img_width).astype(np.int32), img_width - patch_x0_abs)
    patch_height_abs = min(np.round(patch_height_rel * img_height).astype(np.int32), img_height - patch_y0_abs)
    # Image:
    patch = image[patch_y0_abs:(patch_y0_abs+patch_height_abs), patch_x0_abs:(patch_x0_abs+patch_width_abs), :]
    # Bounding boxes:
    patch_x1_rel = patch_x0_rel + patch_width_rel
    patch_y1_rel = patch_y0_rel + patch_height_rel
    remaining_boxes_list = []
    for i in range(bboxes.shape[0]):
        x_center = bboxes[i, 1] + float(bboxes[i, 3]) / 2
        y_center = bboxes[i, 2] + float(bboxes[i, 4]) / 2
        # print([patch_x0_rel, x_center, patch_x1_rel, patch_y0_rel, y_center, patch_y1_rel])
        if patch_x0_rel < x_center < patch_x1_rel and patch_y0_rel < y_center < patch_y1_rel:
            # print('box valid')
            new_box_x0 = (bboxes[i, 1] - patch_x0_rel) / patch_width_rel
            new_box_y0 = (bboxes[i, 2] - patch_y0_rel) / patch_height_rel
            new_box_x1 = (bboxes[i, 1] + bboxes[i, 3] - patch_x0_rel) / patch_width_rel
            new_box_y1 = (bboxes[i, 2] + bboxes[i, 4] - patch_y0_rel) / patch_height_rel
            new_box_x0 = max(new_box_x0, 0)
            new_box_y0 = max(new_box_y0, 0)
            new_box_x1 = min(new_box_x1, 1)
            new_box_y1 = min(new_box_y1, 1)
            new_box_width = new_box_x1 - new_box_x0
            new_box_height = new_box_y1 - new_box_y0
            remaining_boxes_list.append([bboxes[i, 0], new_box_x0, new_box_y0, new_box_width, new_box_height])
        else:
            pass
            # print('box not valid')
    nboxes_remaining = len(remaining_boxes_list)
    remaining_boxes = np.zeros(shape=(nboxes_remaining, 5), dtype=np.float32)
    for i in range(nboxes_remaining):
        remaining_boxes[i, 0] = remaining_boxes_list[i][0]
        remaining_boxes[i, 1] = remaining_boxes_list[i][1]
        remaining_boxes[i, 2] = remaining_boxes_list[i][2]
        remaining_boxes[i, 3] = remaining_boxes_list[i][3]
        remaining_boxes[i, 4] = remaining_boxes_list[i][4]
    return patch, remaining_boxes


def sample_random_patch(image, bboxes, min_scale, max_scale, min_aspect_ratio, max_aspect_ratio):
    # image: (image_height, image_width, 3)
    # bboxes: (nboxes, 5), [class_id, x_min, y_min, width, height]
    img_height, img_width, _ = image.shape
    # Get patch shape:
    patch_x0_rel, patch_y0_rel, patch_width_rel, patch_height_rel = make_patch_shape(img_width, img_height, min_scale, max_scale, min_aspect_ratio, max_aspect_ratio)
    # Extract patch and convert bounding boxes:
    patch, remaining_boxes = sample_patch(image, bboxes, img_width, img_height, patch_x0_rel, patch_y0_rel, patch_width_rel, patch_height_rel)
    return patch, remaining_boxes

def random_crop(image, min_proportion, prob):
    proportion = tf.random_uniform(shape=(), minval=min_proportion, maxval=1)
    # proportion = tf.Print(proportion, [proportion], 'proportion')
    proportion_vec = tf.stack([proportion, proportion, tf.ones(shape=(), dtype=tf.float32)], axis=0)
    original_size = tf.shape(image)
    # original_size = tf.Print(original_size, [original_size], 'original_size')
    new_size = tf.cast(tf.round(tf.cast(original_size, tf.float32) * proportion_vec), tf.int32)
    # new_size = tf.Print(new_size, [new_size], 'new_size')
    crop = tf.random_crop(image, new_size)
    flag = tf.random_uniform(()) < prob
    image = tf.cond(flag, lambda: crop, lambda: tf.identity(image))
    return image

def adjust_contrast(image, factor):
    image = tf.clip_by_value(127.5 + factor * (image - 127.5), 0, 255)
    return image

def random_adjust_contrast(image, factor_lower, factor_upper, prob):
    factor = tf.random_uniform(shape=(), minval=factor_lower, maxval=factor_upper)
    flag = tf.random_uniform(()) < prob
    image = tf.cond(flag, lambda: adjust_contrast(image, factor), lambda: image)
    return image

def adjust_brightness(image, brightness_delta):
    image = tf.clip_by_value(tf.image.adjust_brightness(image, brightness_delta), 0, 255)
    return image

def random_adjust_brightness(image, delta_lower, delta_upper, prob):
    delta_brightness = tf.random_uniform(shape=(), minval=delta_lower, maxval=delta_upper)
    flag = tf.random_uniform(()) < prob
    image = tf.cond(flag, lambda: adjust_brightness(image, delta_brightness), lambda: image)
    return image

def random_adjust_saturation(image, factor_lower, factor_upper, prob):
    factor = tf.random_uniform(shape=(), minval=factor_lower, maxval=factor_upper)
    flag = tf.random_uniform(()) < prob
    image = tf.cond(flag, lambda: tf.image.adjust_saturation(image, factor), lambda: image)
    return image

def random_adjust_hue(image, delta_lower, delta_upper, prob):
    delta_hue = tf.random_uniform(shape=(), minval=delta_lower, maxval=delta_upper)
    flag = tf.random_uniform(()) < prob
    image = tf.cond(flag, lambda: tf.image.adjust_hue(image, delta_hue), lambda: image)
    return image

def convert_to_grayscale(image, prob):
    image_gray = tf.tile(tf.image.rgb_to_grayscale(image), [1, 1, 3])
    flag = tf.random_uniform(()) < prob
    image = tf.cond(flag, lambda: image_gray, lambda: image)
    return image