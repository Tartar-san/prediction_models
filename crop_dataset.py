import os
import argparse
import config
import cv2


def crop_image(image_path, image_size, where_to_save):
    """Crop 768x768 image to img_size*img_size images. Saves new images in format image_name_{x}_{y}"""
    img = cv2.imread(image_path)

    for i in range(768 / image_size):
        for j in range(768 / image_size):
            cropped_img_filename = os.path.basename(image_path)[:-4] + "_".join(["", str(i), str(j)])+".jpg"
            cropped_img_filepath = os.path.join(where_to_save, cropped_img_filename)
            cropped_img = img[i*image_size:(i+1)*image_size, j*image_size:(j+1)*image_size]
            cv2.imwrite(cropped_img_filepath, cropped_img)


def crop_folder(folder_path, image_size, where_to_save):
    """Crop all images in given folder to new size and save in the same folder but on the new path"""
    new_folder_path = os.path.join(where_to_save, os.path.basename(folder_path))
    if not os.path.exists(new_folder_path):
        os.mkdir(new_folder_path)
    else:
        os.rmdir(new_folder_path)
        os.mkdir(new_folder_path)
    for img in os.listdir(folder_path):
        if img[-4:] != ".jpg":
            continue
        else:
            crop_image(os.path.join(folder_path, img), image_size, new_folder_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', type=int, help="Server you use (1 - work, 2 - ucu)", default=1)
    parser.add_argument('--size', type=int, help="Resulting size of cropping. Should be a divider of 768", default=config.working_size)
    parser.add_argument('--result_path', help="Path where to save new dataset")
    args = vars(parser.parse_args())

    if args["server"] == 1:
        paths = config.server1_paths
    elif args["server"] == 2:
        paths = config.server2_paths
    else:
        print("WRONG SERVER TYPE!")
        raise ValueError

    new_size = args["size"]

    if 768 % new_size != 0:
        print("WRONG NEW SIZE")
        raise ValueError

    if not os.path.exists(args["result_path"]):
        os.mkdir(args["result_path"])

    # crop test
    crop_folder(os.path.join(paths["data_root"], config.dataset_paths["test"]), new_size, args["result_path"])

    # crop train
    crop_folder(os.path.join(paths["data_root"], config.dataset_paths["train"]), new_size, args["result_path"])

    # crop train masks
    crop_folder(os.path.join(paths["data_root"], config.dataset_paths["train_masks"]), new_size, args["result_path"])




