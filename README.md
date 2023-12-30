# SMILEtrack

---

This Fork `docker-onnx branch` is an experimental environment to experiment with Docker execution environment and inference with onnxruntime.

```bash
git clone https://github.com/PINTO0309/SMILEtrack && cd SMILEtrack

docker pull docker.io/pinto0309/smiletrack:latest

docker run --rm -it --gpus all \
-v `pwd`:/workdir \
docker.io/pinto0309/smiletrack:latest

cd BoT-SORT

# Tracking test
# - Weights are automatically dunloaded at runtime.
# - Image data sets for verification are not automatically downloaded.
python tools/track.py \
/workdir/BoT-SORT/MOT17Det/train/MOT17-04/img1 \
--default-parameters \
--with-reid \
--benchmark MOT17 \
--eval test \
--fp16 \
--fuse \
--save-frames
```

https://github.com/PINTO0309/SMILEtrack/assets/33194443/93664cca-055d-4717-bcbf-24256bea3640

---


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/smiletrack-similarity-learning-for-multiple/multi-object-tracking-on-mot17)](https://paperswithcode.com/sota/multi-object-tracking-on-mot17?p=smiletrack-similarity-learning-for-multiple)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/smiletrack-similarity-learning-for-multiple/multi-object-tracking-on-mot20-1)](https://paperswithcode.com/sota/multi-object-tracking-on-mot20-1?p=smiletrack-similarity-learning-for-multiple)

This code is based on the implementation of [ByteTrack](https://github.com/ifzhang/ByteTrack), [BoT-SORT](https://github.com/NirAharon/BoT-SORT#bot-sort)

> **SMILEtrack: SiMIlarity LEarning for Multiple Object Tracking**
>
> Preprint will be appearing soon

# 1. Installation

SMILEtrack code is based on [ByteTrack](https://github.com/ifzhang/ByteTrack) and [BoT-SORT](https://github.com/NirAharon/BoT-SORT#bot-sort)

Visit their installation guides for more setup options.

# 2. Download
PRBNet MOT17 weight [link](https://drive.google.com/file/d/1HRjka6Ma7Nrcmzt9FWNQ2ATviNGBuXLC/view?usp=share_link)

PRBNet MOT20 weight [link](https://drive.google.com/file/d/1KyRJNgfApv3m7cHdW7Ekt87pxrs_3ozu/view?usp=share_link)

SLM weight [link](https://drive.google.com/file/d/1RDuVo7jYBkyBR4ngnBaVQUtHL8nAaGaL/view?usp=share_link)

# 3.Data Preparation
Download [MOT17](https://motchallenge.net/data/MOT17/) from the [official website](https://motchallenge.net/). And put them in the following structure:
```
<dataets_dir>
      │
      ├── MOT17
      │      ├── train
      │      └── test
      └——————crowdhuman
      |         └——————Crowdhuman_train
      |         └——————Crowdhuman_val
      |         └——————annotation_train.odgt
      |         └——————annotation_val.odgt
      └——————MOT20
      |        └——————train
      |        └——————test
      └——————Cityscapes
               └——————images
               └——————labels_with_ids


```
# 4.Training PRBNet
Single GPU training
```
cd <prb_dir>
$ python train_aux.py --workers 8 --device 0 --batch-size 4 --data data/mot.yaml --img 1280 1280 --cfg cfg/training/PRB_Series/yolov7-PRB-2PY-e6e-tune-auxpy1.yaml --weights './yolov7-prb-2py-e6e.pt' --name yolov7-prb --hyp data/hyp.scratch.p6.yaml --epochs 100
```
# 5.Training SLM
## Data structure
```
<dataets_dir>
    ├─A
    ├─B
    ├─label
    └─list

```
A: images of t1 phase;

B: images of t2 phase;

label: label maps;

list: contains train.txt, val.txt and test.txt, each file records the image names (XXX.png) in the change detection dataset.

For the more detail of the training setting, you can follow [BIT_CD](https://github.com/justchenhao/BIT_CD) training code.

# 6.Tracking

By submitting the txt files produced in this part to MOTChallenge website and you can get the same results as in the paper.
Tuning the tracking parameters carefully could lead to higher performance. In the paper we apply ByteTrack's calibration.

## Track by detector YOLOX
```
cd <BoT-SORT_dir>
$ python3 tools/track.py <dataets_dir/MOT17> --default-parameters --with-reid --benchmark "MOT17" --eval "test" --fp16 --fuse
$ python3 tools/interpolation.py --txt_path <path_to_track_result>
```
## Track by detector PRBNet
```
cd <BoT-SORT_dir>
$ python3 tools/track_prb.py <dataets_dir/MOT17> --default-parameters --with-reid --benchmark "MOT17" --eval "test" --fp16 --fuse
$ python3 tools/interpolation.py --txt_path <path_to_track_result>
```
# 7.Tracking performance
## Results on MOT17 challenge test set
| Tracker | MOTA | IDF1 | HOTA |
|-------|:-----:|------:|------:|
| SMILEtrack |  81.06  |   80.5 |   65.28    |


## Results on MOT20 challenge test set
| Tracker | MOTA | IDF1 | HOTA |
|-------|:-----:|------:|------:|
| SMILEtrack |  78.19  |   77.53 |   65.28    |

# 8.Acknowledgement
A large part of the codes, ideas and results are borrowed from [PRBNet](https://github.com/pingyang1117/PRBNet_PyTorch), [ByteTrack](https://github.com/ifzhang/ByteTrack), [BoT-SORT](https://github.com/NirAharon/BoT-SORT#bot-sort), [yolov7](https://github.com/WongKinYiu/yolov7), thanks for their excellent work!

