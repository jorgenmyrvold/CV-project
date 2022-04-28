from tops.config import instantiate, LazyConfig
from ssd import utils
from tqdm import tqdm
import matplotlib.pyplot as plt
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
        if isinstance(val, list):
            print(f"{key} \t ", end="")
            for v in val:
                print(f"{v:.2f}\t", end="")
            print("")

        elif isinstance(val, np.ndarray):
            print(f"{key} \t {val}")

        else:
            print(f"{key} \t {val:.3f}")


def get_text_labels(int_labels):
    dataset_labels =  {0: 'background', 1: 'car', 2: 'truck', 3: 'bus', 4: 'motorcycle', 5: 'bicycle', 6: 'scooter', 7: 'person', 8: 'rider'}
    text_labels = []
    for i in int_labels:
        text_labels.append(dataset_labels[i])
    return text_labels


def plot_num_labels(num_labels:dict):
    fig, ax = plt.subplots()
    int_labels_in_dataset = np.array(list(num_labels.keys())).astype(int)
    labels = get_text_labels(int_labels_in_dataset)
    ax.pie(num_labels.values(), labels=labels, autopct='%1.1f%%', shadow=True)
    ax.axis('equal')
    plt.show()


def setBoxColors(bp):
    # Utility function from https://stackoverflow.com/questions/16592222/matplotlib-group-boxplots
    plt.setp(bp['boxes'][0], color='C0')
    plt.setp(bp['caps'][0], color='C0')
    plt.setp(bp['caps'][1], color='C0')
    plt.setp(bp['whiskers'][0], color='C0')
    plt.setp(bp['whiskers'][1], color='C0')
    plt.setp(bp['fliers'][0], markeredgecolor='C0', marker='x')
    plt.setp(bp['medians'][0], color='C0')

    plt.setp(bp['boxes'][1], color='C1')
    plt.setp(bp['caps'][2], color='C1')
    plt.setp(bp['caps'][3], color='C1')
    plt.setp(bp['whiskers'][2], color='C1')
    plt.setp(bp['whiskers'][3], color='C1')
    plt.setp(bp['fliers'][1], markeredgecolor='C1', marker='x')
    plt.setp(bp['medians'][1], color='C1')


def plot_bb_sizes(bb_sizes, y_lim=(0, 425)):
    # bb_sizes = {1:[(w1, h1), (w2 ,h2), ...], 2:...}
    # Want sizes = {1:[(w1, w2, ...), (h1, h2, ...)], 2:...}
    int_labels_in_dataset = np.array(list(bb_sizes.keys())).astype(int)
    text_labels = get_text_labels(int_labels_in_dataset)
    sizes = {}

    for key, size in bb_sizes.items():
        sizes[key] = list(zip(*size))

    fig, ax = plt.subplots()

    bp = plt.boxplot(sizes[1], positions = [1, 2], widths = 0.6)
    setBoxColors(bp)
    bp = plt.boxplot(sizes[2], positions = [4, 5], widths = 0.6)
    setBoxColors(bp)
    bp = plt.boxplot(sizes[3], positions = [7, 8], widths = 0.6)
    setBoxColors(bp)
    # plt.boxplot(sizes[4], positions = [10, 11], widths = 0.6)
    bp = plt.boxplot(sizes[5], positions = [10, 11], widths = 0.6)
    setBoxColors(bp)
    bp = plt.boxplot(sizes[6], positions = [13, 14], widths = 0.6)
    setBoxColors(bp)
    bp = plt.boxplot(sizes[7], positions = [16, 17], widths = 0.6)
    setBoxColors(bp)
    bp = plt.boxplot(sizes[8], positions = [19, 20], widths = 0.6)
    setBoxColors(bp)
    # plt.boxplot(sizes[8], positions = [22, 23], widths = 0.6)

    plt.axhline(y=128, linestyle='dashed', color='C2')
    plt.axhline(y=32, linestyle='dotted', color='C3')
    plt.axhline(y=16, linestyle='dotted', color='C3')
    plt.axhline(y=8, linestyle='dotted', color='C3')
    plt.axhline(y=4, linestyle='dotted', color='C3')
    plt.axhline(y=0, linestyle='solid', color='black')

    hB, = plt.plot([1,1],'C0')
    hR, = plt.plot([1,1],'C1')
    plt.legend((hB, hR),('Width', 'Height'))
    hB.set_visible(False)
    hR.set_visible(False)

    text_labels.pop(3)
    ax.set_xticklabels(text_labels)
    ax.set_xticks([1.5, 4.5, 7.5, 10.5, 13.5, 16.5, 19.5])
    ax.set_ylim(y_lim)
    plt.show()


def plot_ratios(ratios):
    int_labels_in_dataset = np.array(list(ratios.keys())).astype(int)
    text_labels = get_text_labels(int_labels_in_dataset)

    fig, ax = plt.subplots()
    ax.boxplot(ratios.values())

    ax.set_xticklabels(text_labels)
    ax.set_ylim(bottom=0)
    plt.show()
    

def analyze_something(dataloader, cfg):
    img_width = 1024
    img_height = 128

    # Number of lables in the entire dataset
    all_labels = np.array([])
    # Dict containing tuples (w,h) for each bounding box in the corresponding label 
    bb_sizes = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: []}
    pixel_bb_sizes = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: []}
    avg_bb_sizes = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: []}
    pixel_avg_bb_sizes = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: []}
    # Largest bounding boxes
    largest_bb = {'widest':0, 'widest_label':0, 'highest':0, 'highest_label':0}
    # analyze bb ratios
    all_ratios = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: []}
    max_ratio = {1: 0., 2: 0., 3: 0., 4: 0., 5: 0., 6: 0., 7: 0., 8: 0.}
    min_ratio = {1: 999., 2: 999., 3: 999., 4: 999., 5: 999., 6: 999., 7: 999., 8: 999.}
    # (min_height, min_width, max_height, max_width)
    min_max_height_width = {1: [999,999,0,0], 2: [999,999,0,0], 3: [999,999,0,0], 4: [999,999,0,0], 5: [999,999,0,0], 6: [999,999,0,0], 7: [999,999,0,0], 8: [999,999,0,0]}


    for batch in tqdm(dataloader):

        labels = batch["labels"].numpy()
        labels = labels.reshape(labels.shape[1:])
        boxes = batch["boxes"].numpy()
        boxes = boxes.reshape(boxes.shape[1:])
        
        all_labels = np.append(all_labels, batch["labels"])

        for label, bb in zip(labels, boxes):
            height = bb[3] - bb[1]
            width = bb[2] - bb[0]
            pixel_height = height * img_height
            pixel_width = width * img_width
            bb_sizes[label].append((width, height))
            pixel_bb_sizes[label].append((pixel_width, pixel_height))
            
            if pixel_height > largest_bb["highest"]:
                largest_bb["highest"] = pixel_height
                largest_bb["highest_label"] = label
            if pixel_width > largest_bb["widest"]:
                largest_bb["widest"] = pixel_width
                largest_bb["widest_label"] = label
            
            temp_pixel_ratio = pixel_height/pixel_width
            all_ratios[label].append(temp_pixel_ratio)
            if temp_pixel_ratio > max_ratio[label]: max_ratio[label] = temp_pixel_ratio
            if temp_pixel_ratio < min_ratio[label]: min_ratio[label] = temp_pixel_ratio

            if pixel_height < min_max_height_width[label][0]: min_max_height_width[label][0] = pixel_height
            if pixel_height > min_max_height_width[label][2]: min_max_height_width[label][2] = pixel_height
            if pixel_width < min_max_height_width[label][1]: min_max_height_width[label][1] = pixel_width
            if pixel_width > min_max_height_width[label][3]: min_max_height_width[label][3] = pixel_width
            
    unique, counts = np.unique(all_labels, return_counts=True)
    # dict containing number of lables for each lable
    num_labels = dict(zip(unique, counts))

    for key in avg_bb_sizes: 
        sizes = np.array(bb_sizes[key])
        avg_bb_sizes[key] = np.mean(sizes, axis=0)

        pixel_size = np.array(pixel_bb_sizes[key])
        pixel_avg_bb_sizes[key] = np.mean(pixel_size, axis=0)

        try:
            ratio = avg_bb_sizes[key][1] / avg_bb_sizes[key][0]
            pixel_ratio = pixel_avg_bb_sizes[key][1] / pixel_avg_bb_sizes[key][0]
        except IndexError:
            ratio = [np.nan, np.nan]
            pixel_ratio = [np.nan, np.nan]

        avg_bb_sizes[key] = np.append(avg_bb_sizes[key], ratio)
        pixel_avg_bb_sizes[key] = np.append(pixel_avg_bb_sizes[key], pixel_ratio)

    print("\nNUMBER OF LABLES")
    print_dict(num_labels)
    print("\nAVERAGE BOUNDING BOX SIZE (w, h, ratio)")
    print_dict(avg_bb_sizes)
    print("\nPIXEL AVERAGE BOUNDING BOX SIZE (w, h, ratio)")
    print_dict(pixel_avg_bb_sizes)
    print(f"\nMax ratios")
    print_dict(max_ratio)
    print(f"\nMin ratios")
    print_dict(min_ratio)
    print("EXTREME VALUES OF BB: (min_height, min_width, max_height, max_width)")
    print_dict(min_max_height_width)
    print(f'\nWIDEST BB: {largest_bb["widest"]:.2f}\t label {largest_bb["widest_label"]}')
    print(f'HIGHEST BB: {largest_bb["highest"]:.2f}\t label {largest_bb["highest_label"]}')

    plot_num_labels(num_labels)
    # plot_bb_sizes(pixel_bb_sizes)
    # plot_bb_sizes(pixel_bb_sizes, y_lim=(0,20))
    # plot_ratios(all_ratios)
    exit()
    


def main():
    config_path = "configs/tdt4265.py"
    cfg = get_config(config_path)
    dataset_to_analyze = "val"  # "train/val"

    print("Label map is:", cfg.label_map)

    dataloader = get_dataloader(cfg, dataset_to_analyze)
    analyze_something(dataloader, cfg)


if __name__ == '__main__':
    main()
