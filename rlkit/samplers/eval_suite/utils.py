import numpy as np
import cv2
import os

def get_obs_preprocessor(observation_key, additional_keys, desired_goal_key):
    def obs_processor(o):
        obs = o[observation_key]
        for additional_key in additional_keys:
            obs = np.hstack((obs, o[additional_key]))

        return np.hstack((obs, o[desired_goal_key]))
    return obs_processor

def create_blank_image_directories(save_folder, epoch):
    eval_blank_path = f"{save_folder}/epochs/{epoch}/eval_images_blank"
    try:
        os.makedirs(eval_blank_path)
    except:
        print("blank image folders existed already")

def create_real_corner_image_directories(save_folder, epoch):
    real_corners_path = f"{save_folder}/epochs/{epoch}/real_corners_prediction"
    try:
        os.makedirs(real_corners_path)
    except:
        print("real image folders existed already")

def create_real_corner_image_dump_directories(save_folder, prefix, epoch):
    real_corners_path = f"{save_folder}/epochs/{epoch}/real_corners_dump/{prefix}"
    try:
        os.makedirs(real_corners_path)
    except:
        print(prefix, "dump folders existed already")

def save_blank_images(env, save_folder, epoch, step_number, aux_output):
    corner_image, eval_image, cnn_color_image_full, cnn_color_image, cnn_image = env.capture_images(aux_output)
    cv2.imwrite(f'{save_folder}/epochs/{epoch}/eval_images_blank/{str(step_number).zfill(3)}.png', eval_image)

def create_regular_image_directories(save_folder, prefix, epoch):
    cnn_path = f"{save_folder}/epochs/{epoch}/{prefix}/cnn_images"
    cnn_color_path = f"{save_folder}/epochs/{epoch}/{prefix}/cnn_color_images"
    cnn_color_full_path = f"{save_folder}/epochs/{epoch}/{prefix}/cnn_color_full_images"
    corners_path = f"{save_folder}/epochs/{epoch}/{prefix}/corners_images"
    eval_path = f"{save_folder}/epochs/{epoch}/{prefix}/eval_images"
    try:
        os.makedirs(cnn_path)
        os.makedirs(cnn_color_path)
        os.makedirs(cnn_color_full_path)
        os.makedirs(corners_path)
        os.makedirs(eval_path)
    except:
        print("regular folders existed already", prefix, epoch)

def create_base_epoch_directory(save_folder, epoch):
    base_path = f"{save_folder}/epochs/{epoch}"
    try:
        os.makedirs(base_path)
    except:
        print("base epoch folders existed already")

def save_regular_images(env, save_folder, prefix, epoch, step_number, aux_output):
    corner_image, eval_image, cnn_color_image_full, cnn_color_image, cnn_image = env.capture_images(aux_output)
    cv2.imwrite(f'{save_folder}/epochs/{epoch}/{prefix}/corners_images/{str(step_number).zfill(3)}.png', corner_image)
    cv2.imwrite(f'{save_folder}/epochs/{epoch}/{prefix}/eval_images/{str(step_number).zfill(3)}.png', eval_image)
    cv2.imwrite(f'{save_folder}/epochs/{epoch}/{prefix}/cnn_images/{str(step_number).zfill(3)}.png', cnn_image)
    cv2.imwrite(f'{save_folder}/epochs/{epoch}/{prefix}/cnn_color_images/{str(step_number).zfill(3)}.png', cnn_color_image)
    cv2.imwrite(f'{save_folder}/epochs/{epoch}/{prefix}/cnn_color_full_images/{str(step_number).zfill(3)}.png', cnn_color_image_full)