import os
import cv2
import argparse
import config
import numpy as np
import csv
from utils import rle_decode

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', type=int, help="Server you use (1 - work, 2 - ucu)", default=1)
    args = vars(parser.parse_args())
    print(args)
    if args["server"] == 1:
        paths = config.server1_paths
    elif args["server"] == 2:
        paths = config.server2_paths
    else:
        print("WRONG SERVER TYPE!")
        raise ValueError

    masks_path = os.path.join(paths["data_root"], config.dataset_paths["train_masks"])

    if not masks_path:
        os.mkdir(masks_path)

    with open(os.path.join(paths["data_root"], config.dataset_paths["train_ship_segmentations"])) as csv_file:
        reader = csv.reader(csv_file)
        # skip header
        next(reader)
        for i, row in enumerate(reader):
            img_name, lre_string = row
            img_path = os.path.join(masks_path, img_name[:-3] + "npy")
            # print(lre_string)
            if len(lre_string) == 0:
                print("EMPTY!")
                continue

            mask = rle_decode(lre_string)

            print(np.sum(mask))
            np.save(img_path, mask)




