# SMILEtrack

---

This Fork `docker-onnx branch` is an experimental environment to experiment with Docker execution environment and inference with onnxruntime.

- Inference test

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

- onnx export
  - `BoT-SORT` - `fast_reid` - `fastreid` - `modeling` - `meta_arch` - `baseline.py` - `onnx_export=True`

    ```python
    def forward(self, batched_inputs):
      images = self.preprocess_image(batched_inputs, onnx_export=True)
    ```

- Similarity validation

  ||image.1|image.2|
  |:-|:-:|:-:|
  |30 vs 31⬇️|![00030](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/b2249f44-cd26-49da-8796-25e12f2831fe)|![00031](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/030faa0d-b5a3-457e-8402-698f8bfea769)|
  |30 vs 1⬇️|![00030](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/893ed42c-4a63-4779-97e2-2af9ae57a79f)|![1](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/8afb01a8-f7c4-483f-9387-62e59d715693)|
  |31 vs 2⬇️|![00031](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/030faa0d-b5a3-457e-8402-698f8bfea769)|![2](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/c6854b42-25af-42da-b8b0-59f85ee2fb78)|
  |1 vs 2⏫|![1](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/82854902-c63b-4b24-859d-23661fe65f0c)|![2](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/c6854b42-25af-42da-b8b0-59f85ee2fb78)|
  |1 vs 3⏫|![1](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/49f09597-94c8-4130-aa43-b4f3971ed9a7)|![3](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/79ba35d2-88de-4534-9bf5-c1c64d36c279)|
  |1 vs 4⏫|![1](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/8fae11e3-1a46-4907-85b4-f9a9d3257e47)|![4](https://github.com/PINTO0309/PINTO_model_zoo/assets/33194443/c32a10d9-bb67-484f-8483-4c7080e70312)|

  ```bash
  python validation.py
  ```

  |Model|30<br>vs<br>31<br>⬇️|30<br>vs<br>1<br>⬇️|31<br>vs<br>2<br>⬇️|1<br>vs<br>2<br>⏫|1<br>vs<br>3<br>⏫|1<br>vs<br>4<br>⏫|
  |:-|-:|-:|-:|-:|-:|-:|
  |mot17_sbs_S50_NMx3x256x128_post.onnx|0.148|0.046|0.219|0.359|0.611|0.543|
  |mot17_sbs_S50_NMx3x288x128_post.onnx|0.154|0.036|0.223|0.375|0.643|0.562|
  |mot17_sbs_S50_NMx3x320x128_post.onnx|0.093|0.002|0.180|0.386|0.635|0.631|
  |mot17_sbs_S50_NMx3x352x128_post.onnx|0.057|-0.040|0.153|0.366|0.642|0.649|
  |mot17_sbs_S50_NMx3x384x128_post.onnx|0.044|-0.044|0.139|0.359|0.629|0.686|
  |mot20_sbs_S50_NMx3x256x128_post.onnx|0.406|0.318|0.309|0.538|0.727|0.778|
  |mot20_sbs_S50_NMx3x288x128_post.onnx|0.393|0.288|0.324|0.544|0.724|0.770|
  |mot20_sbs_S50_NMx3x320x128_post.onnx|0.372|0.253|0.293|0.543|0.701|0.775|
  |mot20_sbs_S50_NMx3x352x128_post.onnx|0.351|0.243|0.301|0.578|0.695|0.756|
  |mot20_sbs_S50_NMx3x384x128_post.onnx|0.325|0.226|0.289|0.559|0.698|0.757|

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

