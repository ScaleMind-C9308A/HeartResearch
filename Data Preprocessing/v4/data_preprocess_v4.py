import pandas as pd
import scipy.io
import scipy.signal as ss
import pywt
import glob
import matplotlib.pyplot as plt
import scaleogram as scg
from scaleogram import cws
import numpy as np
from math import *
import os
import argparse
from datetime import datetime
from tqdm import tqdm

# from modules.custom_cws import cws as cus_cws

# print(os.getcwd())

for i in range(2):
    os.chdir("..")

# print(os.getcwd())

main_data_dir = os.getcwd() + "/Data set"
# print(os.listdir(main_data_dir))

data_mat_files = glob.glob(main_data_dir + "/TrainingSet*/*")
print("data_mat_files: {}".format(len(data_mat_files)))
label_path = main_data_dir + "/Label.csv"
label_df = pd.read_csv(label_path)
# print("label_df: {}".format(label_df.shape))
label_df.drop(['Second_label', 'Third_label'], axis=1)

def filter_mapping(filter: str = "med", x: np.array = None):
    if filter == "med":
        return ss.medfilt(x)

def data_extract(signal_paths: list = data_mat_files, start_point: int = 300, 
                filter:str = "med", save_path:str = main_data_dir + "/v4_data",
                save_segment: bool = True, save_filter:bool = True, save_scale:bool = True,
                scale:int = 500, img_size: tuple = (256, 512), seg_len:int = 1600):
    
    scg.set_default_wavelet('morl')
    px = 1/plt.rcParams['figure.dpi']
    lead = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
    
    if not signal_paths:
        raise Exception("signal_paths arg cannot be None")
    else:
        signal_paths.sort()
    
    if save_path:
        if not os.path.exists(save_path):
            os.mkdir(save_path)

    if save_segment:
        if save_path:
            if seg_len:
                segment_dir = save_path + f"/segment_{seg_len}"
                if not os.path.exists(segment_dir):
                    os.mkdir(segment_dir)
                if filter:
                    segment_dir = segment_dir + f"/{filter}"
                    if not os.path.exists(segment_dir):
                        os.mkdir(segment_dir)
            else:
                raise Exception("seg_len arg cannot be None as save_segment arg is True")
        else:
            raise Exception("save_path arg cannot be None as save_segment arg is True")
    
    if save_filter:
        if save_path:
            if filter:
                filter_dir = save_path + f"/filter_{filter}"
                if not os.path.exists(filter_dir):
                    os.mkdir(filter_dir)
            else:
                raise Exception("filter arg cannot be None as save_filter arg is True")
        else:
            raise Exception("save_path arg cannot be None as save_filter arg is True")
    
    if save_scale:
        for x in [save_filter, img_size, save_segment, scale, start_point, save_path]:
            if x is None:
                raise Exception(f"{x} arg cannot be None as save_scale arg is True")
        if save_path:
            scale_dir = save_path + f"/{filter}_scaleogram_h{img_size[0]}_w{img_size[1]}_seglen{seg_len}_scl{scale}"
            if not os.path.exists(scale_dir):
                os.mkdir(scale_dir)
        else:
            raise Exception("save_path arg cannot be None as save_scale arg is True")
    
    for sp_idx in tqdm(range(len(signal_paths))):
        signal_path = signal_paths[sp_idx]
        signal_filename = signal_path.split("\\")[-1].split(".")[0]

        filter_save_path = filter_dir + "/{}_lead{}.mat" # filname _ lead
        segment_save_path = segment_dir + "/{}_lead{}_seg{}.mat" #filename _ lead _ seg
        scaleogram_save_path = scale_dir + "/{}_lead{}_seg{}"

        signal = scipy.io.loadmat(signal_path)['ECG'][0][0][2][:,start_point:]
        sig_len = signal.shape[1]
        # print("sig_len: {}".format(sig_len))
        for idx, lead in enumerate(signal):
            # print("lead: {}".format(lead.shape))
            if filter:
                filter_lead = filter_mapping(filter = filter, x = lead)
            if save_segment:
                seg_num = ceil(sig_len/seg_len)
                # print("seg_num: {}".format(seg_num))
                for w in range(1,seg_num+1):
                    index = floor((sig_len-seg_len)/(seg_num-1)*(w-1))
                    segment = filter_lead[index:index+seg_len]
                    # print(segment.shape)
                    segment_save_data = {"ECG_segment": segment}
                    scipy.io.savemat(segment_save_path.format(signal_filename, idx+1, w), segment_save_data)

                    if save_scale:
                        fig, ax = cws(segment, 
                                scales=scg.periods2scales(np.arange(1, scale)),
                                figsize=(img_size[1]*px, img_size[0]*px), coi = False)
                        plt.axis('off')
                        plt.savefig(scaleogram_save_path.format(signal_filename, idx, w), 
                                    bbox_inches='tight', 
                                    # format = 'jpg', 
                                    pad_inches = 0)
                        fig.clear()
                        plt.close(fig)
                        ax.clear()
                        plt.cla()
                        plt.close("all")

                    # break
            # break
        # break


if __name__ == '__main__':
    print("__main__")

    parser = argparse.ArgumentParser()
    parser.add_argument("--start_point", type=int, default=300,
        help="index cutting signal")
    parser.add_argument("--filter", type=str, default="med",
        help="Type of filter")
    parser.add_argument("--save_path", type=str, default=main_data_dir + "/v4_data",
        help="Save path dir")
    parser.add_argument("--save_segment", type=bool, default=True,
        help="True if you want to save segmented file")
    parser.add_argument("--save_filter", type=bool, default=True, 
        help="True if you want to save filterd file")
    parser.add_argument("--save_scale", type=bool, default=True, 
        help="True if you want to save scaled file")
    parser.add_argument("--scale", type=int, default=500, 
        help="Number of scale")
    parser.add_argument("--img_size", type=tuple, default=(256, 512), 
        help="Size of output scaleogram")
    parser.add_argument("--seg_len", type=int, default=1600, 
        help="Length of segmented signal")
    args = parser.parse_args()

    print("datetime - start: {}".format(datetime.now()))
    data_extract(start_point=args.start_point, filter=args.filter, save_path = args.save_path, 
                    save_segment=args.save_segment, save_scale=args.save_scale, save_filter=args.save_filter,
                    scale=args.scale, img_size=args.img_size, seg_len=args.seg_len)
    print("datetime - end: {}".format(datetime.now()))