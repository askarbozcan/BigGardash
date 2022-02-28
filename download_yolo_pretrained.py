from genericpath import exists
import os
import glob
from os.path import join as pjoin
import click
from tqdm import tqdm
import requests

URLS = {
    "yolov4": {
        "data": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/coco.data",
        "names": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/coco.names",
        "weights": "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights",
        "cfg": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg",
    },

    "yolov4-tiny-216": {
        "data": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/coco.data",
        "names": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/coco.names",
        "weights": "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights",
        "cfg":"https://gist.githubusercontent.com/askarbozcan/d6beb60243dfe6ebd9b25c43a5092061/raw/21402c1ae40621f2e23526251630251fe09d411a/yolov4-tiny.cfg",
    },
    "yolov4-tiny":{
        "data": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/coco.data",
        "names": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/coco.names",
        "cfg": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg",
        "weights": "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights"
    }
}


def dl_file(url, save_path):
    response = requests.get(url, stream=True)
    total_size_in_bytes= int(response.headers.get('content-length', 0))
    block_size = 1024 #1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

    with open(save_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

@click.command()
@click.option("-f","--models_path", help="Folder where YOLO models should be downloaded.", required=True)
@click.option("-n","--pretrained_name", help="Pretrained model name")
def run(models_path, pretrained_name=None):

    if pretrained_name not in URLS and pretrained_name is not None:
        print("Wrong model name. Please choose one of the models: ")
        for k in URLS:
            print(f"*\"{k}\"")
        
        return
    
    if pretrained_name is None:
        models_to_dl = list(URLS.keys())
    else:
        models_to_dl = [pretrained_name]
    

    for m in models_to_dl:
        os.makedirs(pjoin(models_path, m), exist_ok=True)
        for kk in URLS[m]:
            dl_file(URLS[m][kk], pjoin(models_path, m, os.path.basename(URLS[m][kk])))

    print(f"Downloaded {models_to_dl}")



if __name__== "__main__":
    run()