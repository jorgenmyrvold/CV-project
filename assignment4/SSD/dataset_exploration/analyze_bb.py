from tops.config import instantiate, LazyConfig
from ssd import utils
from tqdm import tqdm
import numpy as np
np.set_printoptions(precision=4, suppress=True)


def get_config(config_path):
    cfg = LazyConfig.load(config_path)
    cfg.train.batch_size = 1
    return cfg


def get_dataloader(cfg, dataset_to_visualize):
    if dataset_to_visualize == "train":
        # Remove GroundTruthBoxesToAnchors transform
        cfg.data_train.dataset.transform.transforms = cfg.data_train.dataset.transform.transforms[:-1]
        data_loader = instantiate(cfg.data_train.dataloader)
    else:
        cfg.data_val.dataloader.collate_fn = utils.batch_collate
        data_loader = instantiate(cfg.data_val.dataloader)

    return data_loader

def print_dict(d):
    for key, val in d.items():
        print(f"{key} \t {val}")


def analyze_something(dataloader, cfg):
    img_width = 1024
    img_height = 128

    # Number of lables in the entire dataset
    all_labels = np.array([])
    # Dict containing tuples (w,h) for each bounding box in the corresponding label 
    bb_sizes = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: []}
    scaled_bb_sizes = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: []}
    avg_bb_sizes = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: []}
    scaled_avg_bb_sizes = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: []}


    for batch in tqdm(dataloader):

        labels = batch["labels"].numpy()
        labels = labels.reshape(labels.shape[1:])
        boxes = batch["boxes"].numpy()
        boxes = boxes.reshape(boxes.shape[1:])
        
        all_labels = np.append(all_labels, batch["labels"])

        for label, bb in zip(labels, boxes):
            height = bb[3] - bb[1]
            width = bb[2] - bb[0]
            bb_sizes[label].append((width, height))
            scaled_bb_sizes[label].append((width * img_width, height * img_height))

    unique, counts = np.unique(all_labels, return_counts=True)
    # dict containing number of lables for each lable
    num_labels = dict(zip(unique, counts))

    for key in avg_bb_sizes: 
        sizes = np.array(bb_sizes[key])
        avg_bb_sizes[key] = np.mean(sizes, axis=0)

        scaled_size = np.array(scaled_bb_sizes[key])
        scaled_avg_bb_sizes[key] = np.mean(scaled_size, axis=0)

        try:
            ratio = avg_bb_sizes[key][1] / avg_bb_sizes[key][0]
            scaled_ratio = scaled_avg_bb_sizes[key][1] / scaled_avg_bb_sizes[key][0]
        except IndexError:
            ratio = [np.nan, np.nan]
            scaled_ratio = [np.nan, np.nan]

        avg_bb_sizes[key] = np.append(avg_bb_sizes[key], ratio)
        scaled_avg_bb_sizes[key] = np.append(scaled_avg_bb_sizes[key], scaled_ratio)

    print("\nNUMBER OF LABLES")
    print_dict(num_labels)
    print("\nAVERAGE BOUNDING BOX SIZE (w, h, ratio)")
    print_dict(avg_bb_sizes)
    print("\nSCALED AVERAGE BOUNDING BOX SIZE (w, h, ratio)")
    print_dict(scaled_avg_bb_sizes)


    exit()
    


def main():
    config_path = "configs/tdt4265.py"
    cfg = get_config(config_path)
    dataset_to_analyze = "train"  # or "val"

    print("Label map is:", cfg.label_map)

    dataloader = get_dataloader(cfg, dataset_to_analyze)
    analyze_something(dataloader, cfg)


if __name__ == '__main__':
    main()
