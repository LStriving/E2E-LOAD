# plot ground truth and prediction array
import numpy as np
import os
import pickle as pkl
import argparse
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm

new_name_replace = [
    "Idle",
    "Corneal incision by 15 degree keratome",
    "Corneal incision by 3.2 mm keratome",
    "Carbachol injection",
    "OVDs injection",
    "Gonioscopy",
    "Goniotomy",
    "OVDs irrigation/aspiration",
    "Would closure"
]


def load_pkl(path:str='./tmp'):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    elif not os.path.isdir(path): # file
        with open(path,'rb') as f:
            data = pkl.load(f)
            yield data, os.path.basename(path)
    else: # dir
        files = os.listdir(path)
        files = [file for file in files if file.endswith('.pkl')]
        
        for file in files:
            with open(os.path.join(path, file), 'rb') as f:
                data = pkl.load(f)
                yield data, file

def process(x):
    """softmax + argmax
    """
    x = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    x = np.argmax(x, axis=1)
    return x

def extend(x):
    """extend the array to 200 frames
    """
    x = np.expand_dims(x, axis=1)
    x = np.repeat(x, 200, axis=1)
    return x

def plot(data, save_path):
    """plot ground truth and prediction array
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for vid in tqdm(data['gt'].keys()):
        if vid not in data['pre'].keys():
            print(f'Error: {vid} not in prediction')
            continue
        # get gt and pred
        gt = data['gt'][vid]
        pred =  data['pre'][vid]
        # process prediction
        gt, pred = process(gt), process(pred)
        # extend the array to 200 frames for plotting
        gt, pred = extend(gt), extend(pred)

        # set the color map
        color_list = sns.color_palette("husl", 9)
        color_list[-1] = (0.84765625,0.84765625,0.84765625)
        color_list = [color_list[-1],*color_list[:-1]]
        color_list = [*color_list[0:2], color_list[-1], *color_list[2:-1]]
        cmap = matplotlib.colors.ListedColormap(color_list)
        
        # set theme
        custom_params = {"axes.spines.right": False, "axes.spines.top": False,"axes.spines.left": False, "axes.spines.bottom": False}
        sns.set_theme(style="ticks", rc=custom_params)

        # plot
        fig, axs = plt.subplots(2, 1, figsize=(10, 4),sharex=True)
        fig.suptitle(f'Comparison of Ground Truth and Prediction of video {vid}')
        fig.subplots_adjust(wspace=0.6)
        axs[0].imshow(gt.T, cmap=cmap)
        axs[0].set_title(f'Ground Truth')
        axs[1].imshow(pred.T, cmap=cmap)
        axs[1].set_title(f'Prediction')


        # Create a custom legend to map colors to action categories
        legend_patches = [plt.Rectangle((0,0),1,1, color=color_list[i]) for i in range(9)]
        # swap the legend order: move the last second to the second and remain the others
        # legend_patches = [legend_patches[0], legend_patches[-2], *legend_patches[1:-2], legend_patches[-1]] 
        # plt.legend(legend_patches, new_name_replace, loc='lower right', prop={'size': 12})
        save_path = save_path.replace('.png', f'_{vid}.png')
        plt.savefig(save_path)
        plt.close()

def plot_2d(data, save_path):
    """plot ground truth and prediction array in 2D"""
    dir_path = os.path.dirname(save_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    for vid in tqdm(data['gt'].keys()):
        if vid not in data['pre'].keys():
            print(f'Error: {vid} not in prediction')
            continue
        # get gt and pred
        gt = data['gt'][vid]
        pred =  data['pre'][vid]

        # set theme
        custom_params = {"axes.spines.right": False, "axes.spines.top": False,"axes.spines.left": False, "axes.spines.bottom": False}
        sns.set_theme(style="ticks", rc=custom_params)
        # plot
        fig, axs = plt.subplots(2, 1, figsize=(10, 4),sharex=True)
        fig.suptitle(f'Comparison of Ground Truth and Prediction of video {vid}')
        fig.subplots_adjust(wspace=0.6)

        sns.heatmap(gt.T, ax=axs[0], cmap='Blues')
        axs[0].xaxis.set_visible(False)
        axs[0].set_title(f'Ground Truth')
        axs[0].set_yticklabels(axs[0].get_yticklabels(), rotation=0)

        sns.heatmap(pred.T, ax=axs[1], cmap='Blues')
        axs[1].set_title(f'Prediction')
        axs[1].xaxis.set_visible(False)
        axs[1].set_yticklabels(axs[0].get_yticklabels(), rotation=0)
        pic_path = save_path.replace('.png', f'_{vid}.png')
        plt.savefig(pic_path)
        plt.close()



def main(args):
    save_dir = args.save_path
    for data, filename in load_pkl(args.path):
        save_path = os.path.join(save_dir, filename.split(".")[0]+'.png')
        if args.dim == 1:
            plot(data, save_path)
        elif args.dim == 2:
            plot_2d(data, save_path)
        else:
            raise ValueError("dim should be 1 or 2")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./tmp/epoch-8.pkl')
    parser.add_argument('--save_path', type=str, default='./tmp/visual_pic/prediction')
    parser.add_argument("--dim", type=int,help="1: 1D, 2: 2D",default=2)
    args = parser.parse_args()
    main(args)
