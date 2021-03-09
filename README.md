<a href="https://apps.apple.com/app/id1452689527" target="_blank">
![image](https://user-images.githubusercontent.com/35050296/110505082-97085c00-80dc-11eb-8174-7b45270e4a28.png)
<img src="https://user-images.githubusercontent.com/26833433/98699617-a1595a00-2377-11eb-8145-fc674eb9b1a7.jpg" width="1000"></a>
&nbsp

![CI CPU testing](https://github.com/ultralytics/yolov5/workflows/CI%20CPU%20testing/badge.svg)

This repository represents Ultralytics open-source research into future object detection methods, and incorporates our lessons learned and best practices evolved over training thousands of models on custom client datasets with our previous YOLO repository https://github.com/ultralytics/yolov3. **All code and models are under active development, and are subject to modification or deletion without notice.** Use at your own risk.

<img src="https://user-images.githubusercontent.com/26833433/90187293-6773ba00-dd6e-11ea-8f90-cd94afc0427f.png" width="1000">** GPU Speed measures end-to-end time per image averaged over 5000 COCO val2017 images using a V100 GPU with batch size 32, and includes image preprocessing, PyTorch FP16 inference, postprocessing and NMS. EfficientDet data from [google/automl](https://github.com/google/automl) at batch size 8.

- **August 13, 2020**: [v3.0 release](https://github.com/ultralytics/yolov5/releases/tag/v3.0): nn.Hardswish() activations, data autodownload, native AMP.
- **July 23, 2020**: [v2.0 release](https://github.com/ultralytics/yolov5/releases/tag/v2.0): improved model definition, training and mAP.
- **June 22, 2020**: [PANet](https://arxiv.org/abs/1803.01534) updates: new heads, reduced parameters, improved speed and mAP [364fcfd](https://github.com/ultralytics/yolov5/commit/364fcfd7dba53f46edd4f04c037a039c0a287972).
- **June 19, 2020**: [FP16](https://pytorch.org/docs/stable/nn.html#torch.nn.Module.half) as new default for smaller checkpoints and faster inference [d4c6674](https://github.com/ultralytics/yolov5/commit/d4c6674c98e19df4c40e33a777610a18d1961145).
- **June 9, 2020**: [CSP](https://github.com/WongKinYiu/CrossStagePartialNetworks) updates: improved speed, size, and accuracy (credit to @WongKinYiu for CSP).
- **May 27, 2020**: Public release. YOLOv5 models are SOTA among all known YOLO implementations.


## Pretrained Checkpoints

| Model | AP<sup>val</sup> | AP<sup>test</sup> | AP<sub>50</sub> | Speed<sub>GPU</sub> | FPS<sub>GPU</sub> || params | FLOPS |
|---------- |------ |------ |------ | -------- | ------| ------ |------  |  :------: |
| [YOLOv5s](https://github.com/ultralytics/yolov5/releases)    | 37.0     | 37.0     | 56.2     | **2.4ms** | **416** || 7.5M   | 13.2B
| [YOLOv5m](https://github.com/ultralytics/yolov5/releases)    | 44.3     | 44.3     | 63.2     | 3.4ms     | 294     || 21.8M  | 39.4B
| [YOLOv5l](https://github.com/ultralytics/yolov5/releases)    | 47.7     | 47.7     | 66.5     | 4.4ms     | 227     || 47.8M  | 88.1B
| [YOLOv5x](https://github.com/ultralytics/yolov5/releases)    | **49.2** | **49.2** | **67.7** | 6.9ms     | 145     || 89.0M  | 166.4B
| | | | | | || |
| [YOLOv5x](https://github.com/ultralytics/yolov5/releases) + TTA|**50.8**| **50.8** | **68.9** | 25.5ms    | 39      || 89.0M  | 354.3B
| | | | | | || |
| [YOLOv3-SPP](https://github.com/ultralytics/yolov5/releases) | 45.6     | 45.5     | 65.2     | 4.5ms     | 222     || 63.0M  | 118.0B

** AP<sup>test</sup> denotes COCO [test-dev2017](http://cocodataset.org/#upload) server results, all other AP results denote val2017 accuracy.  
** All AP numbers are for single-model single-scale without ensemble or TTA. **Reproduce mAP** by `python test.py --data coco.yaml --img 640 --conf 0.001 --iou 0.65`  
** Speed<sub>GPU</sub> averaged over 5000 COCO val2017 images using a GCP [n1-standard-16](https://cloud.google.com/compute/docs/machine-types#n1_standard_machine_types) V100 instance, and includes image preprocessing, FP16 inference, postprocessing and NMS. NMS is 1-2ms/img.  **Reproduce speed** by `python test.py --data coco.yaml --img 640 --conf 0.25 --iou 0.45`  
** All checkpoints are trained to 300 epochs with default settings and hyperparameters (no autoaugmentation). 
** Test Time Augmentation ([TTA](https://github.com/ultralytics/yolov5/issues/303)) runs at 3 image sizes. **Reproduce TTA** by `python test.py --data coco.yaml --img 832 --iou 0.65 --augment` 

## Requirements

Python 3.8 or later with all [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) dependencies installed, including `torch>=1.7`. To install run:
```bash
$ pip install -r requirements.txt
```


## Tutorials

* [Train Custom Data](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)&nbsp; 🚀 RECOMMENDED
* [Weights & Biases Logging](https://github.com/ultralytics/yolov5/issues/1289)&nbsp; 🌟 NEW
* [Multi-GPU Training](https://github.com/ultralytics/yolov5/issues/475)
* [PyTorch Hub](https://github.com/ultralytics/yolov5/issues/36)&nbsp; ⭐ NEW
* [ONNX and TorchScript Export](https://github.com/ultralytics/yolov5/issues/251)
* [Test-Time Augmentation (TTA)](https://github.com/ultralytics/yolov5/issues/303)
* [Model Ensembling](https://github.com/ultralytics/yolov5/issues/318)
* [Model Pruning/Sparsity](https://github.com/ultralytics/yolov5/issues/304)
* [Hyperparameter Evolution](https://github.com/ultralytics/yolov5/issues/607)
* [Transfer Learning with Frozen Layers](https://github.com/ultralytics/yolov5/issues/1314)&nbsp; ⭐ NEW
* [TensorRT Deployment](https://github.com/wang-xinyu/tensorrtx)


## Environments

YOLOv5 may be run in any of the following up-to-date verified environments (with all dependencies including [CUDA](https://developer.nvidia.com/cuda)/[CUDNN](https://developer.nvidia.com/cudnn), [Python](https://www.python.org/) and [PyTorch](https://pytorch.org/) preinstalled):

- **Google Colab Notebook** with free GPU: <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
- **Kaggle Notebook** with free GPU: [https://www.kaggle.com/ultralytics/yolov5](https://www.kaggle.com/ultralytics/yolov5)
- **Google Cloud** Deep Learning VM. See [GCP Quickstart Guide](https://github.com/ultralytics/yolov5/wiki/GCP-Quickstart) 
- **Docker Image** https://hub.docker.com/r/ultralytics/yolov5. See [Docker Quickstart Guide](https://github.com/ultralytics/yolov5/wiki/Docker-Quickstart) ![Docker Pulls](https://img.shields.io/docker/pulls/ultralytics/yolov5?logo=docker)


## Inference

detect.py runs inference on a variety of sources, downloading models automatically from the [latest YOLOv5 release](https://github.com/ultralytics/yolov5/releases) and saving results to `runs/detect`.
```bash
$ python detect.py --source 0  # webcam
                            file.jpg  # image 
                            file.mp4  # video
                            path/  # directory
                            path/*.jpg  # glob
                            rtsp://170.93.143.139/rtplive/470011e600ef003a004ee33696235daa  # rtsp stream
                            rtmp://192.168.1.105/live/test  # rtmp stream
                            http://112.50.243.8/PLTV/88888888/224/3221225900/1.m3u8  # http stream
```

To run inference on example images in `data/images`:
```bash
$ python detect.py --source data/images --weights yolov5s.pt --conf 0.25

Namespace(agnostic_nms=False, augment=False, classes=None, conf_thres=0.25, device='', img_size=640, iou_thres=0.45, save_conf=False, save_dir='runs/detect', save_txt=False, source='data/images/', update=False, view_img=False, weights=['yolov5s.pt'])
Using torch 1.7.0+cu101 CUDA:0 (Tesla V100-SXM2-16GB, 16130MB)

Downloading https://github.com/ultralytics/yolov5/releases/download/v3.1/yolov5s.pt to yolov5s.pt... 100%|██████████████| 14.5M/14.5M [00:00<00:00, 21.3MB/s]

Fusing layers... 
Model Summary: 232 layers, 7459581 parameters, 0 gradients
image 1/2 data/images/bus.jpg: 640x480 4 persons, 1 buss, 1 skateboards, Done. (0.012s)
image 2/2 data/images/zidane.jpg: 384x640 2 persons, 2 ties, Done. (0.012s)
Results saved to runs/detect/exp
Done. (0.113s)
```
<img src="https://user-images.githubusercontent.com/26833433/97107365-685a8d80-16c7-11eb-8c2e-83aac701d8b9.jpeg" width="500">  

### PyTorch Hub

To run **batched inference** with YOLOv5 and [PyTorch Hub](https://github.com/ultralytics/yolov5/issues/36):
```python
import torch
from PIL import Image

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).autoshape()  # for PIL/cv2/np inputs and NMS

# Images
img1 = Image.open('zidane.jpg')
img2 = Image.open('bus.jpg')
imgs = [img1, img2]  # batched list of images

# Inference
prediction = model(imgs, size=640)  # includes NMS
```


## Training

Download [COCO](https://github.com/ultralytics/yolov5/blob/master/data/scripts/get_coco.sh) and run command below. Training times for YOLOv5s/m/l/x are 2/4/6/8 days on a single V100 (multi-GPU times faster). Use the largest `--batch-size` your GPU allows (batch sizes shown for 16 GB devices).
```bash
$ python train.py --data coco.yaml --cfg yolov5s.yaml --weights '' --batch-size 64
                                         yolov5m                                40
                                         yolov5l                                24
                                         yolov5x                                16
```
<img src="https://user-images.githubusercontent.com/26833433/90222759-949d8800-ddc1-11ea-9fa1-1c97eed2b963.png" width="900">


## Citation

[![DOI](https://zenodo.org/badge/264818686.svg)](https://zenodo.org/badge/latestdoi/264818686)


## About Us

Ultralytics is a U.S.-based particle physics and AI startup with over 6 years of expertise supporting government, academic and business clients. We offer a wide range of vision AI services, spanning from simple expert advice up to delivery of fully customized, end-to-end production solutions, including:
- **Cloud-based AI** systems operating on **hundreds of HD video streams in realtime.**
- **Edge AI** integrated into custom iOS and Android apps for realtime **30 FPS video inference.**
- **Custom data training**, hyperparameter evolution, and model exportation to any destination.

For business inquiries and professional support requests please visit us at https://www.ultralytics.com. 


## Contact

**Issues should be raised directly in the repository.** For business inquiries or professional support requests please visit https://www.ultralytics.com or email Glenn Jocher at glenn.jocher@ultralytics.com. 


# Tutorial - Teste de Vídeo com YoloV5

This repo implements [SSD (Single Shot MultiBox Detector)](https://arxiv.org/abs/1512.02325). The implementation is heavily influenced by the projects [ssd.pytorch](https://github.com/amdegroot/ssd.pytorch) and [Detectron](https://github.com/facebookresearch/Detectron).
The design goal is modularity and extensibility.

Currently, it has MobileNetV1, MobileNetV2, and VGG based SSD/SSD-Lite implementations. 

It also has out-of-box support for retraining on Google Open Images dataset.



![gif](https://github.com/PedroFilhoEng/smart-city-canaa/blob/a12c3ce0f307e35bd8feabf0fbddbcf2db9461be/Fazenda_YOLOV5.gif)



## Dependencies
1. Python 3.6+
2. OpenCV
3. Pytorch 1.0 or Pytorch 0.4+
4. Caffe2
5. Pandas
6. Boto3 if you want to train models on the Google OpenImages Dataset.

## Run the demo
### Run the live MobilenetV1 SSD demo

```bash
wget -P models https://storage.googleapis.com/models-hao/mobilenet-v1-ssd-mp-0_675.pth
wget -P models https://storage.googleapis.com/models-hao/voc-model-labels.txt
python run_ssd_live_demo.py mb1-ssd models/mobilenet-v1-ssd-mp-0_675.pth models/voc-model-labels.txt 
```
### Run the live demo in Caffe2

```bash
wget -P models https://storage.googleapis.com/models-hao/mobilenet_v1_ssd_caffe2/mobilenet-v1-ssd_init_net.pb
wget -P models https://storage.googleapis.com/models-hao/mobilenet_v1_ssd_caffe2/mobilenet-v1-ssd_predict_net.pb
python run_ssd_live_caffe2.py models/mobilenet-v1-ssd_init_net.pb models/mobilenet-v1-ssd_predict_net.pb models/voc-model-labels.txt 
```

You can see a decent speed boost by using Caffe2.

### Run the live MobileNetV2 SSD Lite demo

```bash
wget -P models https://storage.googleapis.com/models-hao/mb2-ssd-lite-mp-0_686.pth
wget -P models https://storage.googleapis.com/models-hao/voc-model-labels.txt
python run_ssd_live_demo.py mb2-ssd-lite models/mb2-ssd-lite-mp-0_686.pth models/voc-model-labels.txt 
```

The above MobileNetV2 SSD-Lite model is not ONNX-Compatible, as it uses Relu6 which is not supported by ONNX.
The code supports the ONNX-Compatible version. Once I have trained a good enough MobileNetV2 model with Relu, I will upload
the corresponding Pytorch and Caffe2 models.

You may notice MobileNetV2 SSD/SSD-Lite is slower than MobileNetV1 SSD/Lite on PC. However, MobileNetV2 is faster on mobile devices.

## Pretrained Models

### Mobilenet V1 SSD

URL: https://storage.googleapis.com/models-hao/mobilenet-v1-ssd-mp-0_675.pth

```
Average Precision Per-class:
aeroplane: 0.6742489426027927
bicycle: 0.7913672875238116
bird: 0.612096015101108
boat: 0.5616407126931772
bottle: 0.3471259064860268
bus: 0.7742298893362103
car: 0.7284171192326804
cat: 0.8360675520354323
chair: 0.5142295855384792
cow: 0.6244090341627014
diningtable: 0.7060035669312754
dog: 0.7849252606216821
horse: 0.8202146617282785
motorbike: 0.793578272243471
person: 0.7042670984734087
pottedplant: 0.40257147509774405
sheep: 0.6071252282334352
sofa: 0.7549120254763918
train: 0.8270992920206008
tvmonitor: 0.6459903029666852

Average Precision Across All Classes:0.6755
```

### MobileNetV2 SSD-Lite

URL: https://storage.googleapis.com/models-hao/mb2-ssd-lite-mp-0_686.pth

```
Average Precision Per-class:
aeroplane: 0.6973327307871002
bicycle: 0.7823755921687233
bird: 0.6342429230125619
boat: 0.5478160937380846
bottle: 0.3564069147093762
bus: 0.7882037885117419
car: 0.7444122242934775
cat: 0.8198865557991936
chair: 0.5378973422880109
cow: 0.6186076149254742
diningtable: 0.7369559500950861
dog: 0.7848265495754562
horse: 0.8222948787839229
motorbike: 0.8057808854619948
person: 0.7176976451996411
pottedplant: 0.42802932547480066
sheep: 0.6259124005994047
sofa: 0.7840368059271103
train: 0.8331588002612781
tvmonitor: 0.6555051795079904
Average Precision Across All Classes:0.6860690100560214
```

The code to re-produce the model:

```bash
wget -P models https://storage.googleapis.com/models-hao/mb2-imagenet-71_8.pth
python train_ssd.py --dataset_type voc  --datasets ~/data/VOC0712/VOC2007 ~/data/VOC0712/VOC2012 --validation_dataset ~/data/VOC0712/test/VOC2007/ --net mb2-ssd-lite --base_net models/mb2-imagenet-71_8.pth  --scheduler cosine --lr 0.01 --t_max 200 --validation_epochs 5 --num_epochs 200
```

### VGG SSD

URL: https://storage.googleapis.com/models-hao/vgg16-ssd-mp-0_7726.pth


```
Average Precision Per-class:
aeroplane: 0.7957406334737802
bicycle: 0.8305351156180996
bird: 0.7570969203281721
boat: 0.7043869846367731
bottle: 0.5151666571756393
bus: 0.8375121237865507
car: 0.8581508869699901
cat: 0.8696185705648963
chair: 0.6165431194526735
cow: 0.8066422244852381
diningtable: 0.7629391213959706
dog: 0.8444541531856452
horse: 0.8691922094815812
motorbike: 0.8496564646906418
person: 0.793785185549561
pottedplant: 0.5233462463152305
sheep: 0.7786762429478917
sofa: 0.8024887701948746
train: 0.8713861172265407
tvmonitor: 0.7650514925384194
Average Precision Across All Classes:0.7726184620009084
```

The code to re-produce the model:

```bash
wget -P models https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
python train_ssd.py --datasets ~/data/VOC0712/VOC2007/ ~/data/VOC0712/VOC2012/ --validation_dataset ~/data/VOC0712/test/VOC2007/ --net vgg16-ssd --base_net models/vgg16_reducedfc.pth  --batch_size 24 --num_epochs 200 --scheduler "multi-step” —-milestones “120,160”
```
## Training

```bash
wget -P models https://storage.googleapis.com/models-hao/mobilenet_v1_with_relu_69_5.pth
python train_ssd.py --datasets ~/data/VOC0712/VOC2007/ ~/data/VOC0712/VOC2012/ --validation_dataset ~/data/VOC0712/test/VOC2007/ --net mb1-ssd --base_net models/mobilenet_v1_with_relu_69_5.pth  --batch_size 24 --num_epochs 200 --scheduler cosine --lr 0.01 --t_max 200
```


The dataset path is the parent directory of the folders: Annotations, ImageSets, JPEGImages, SegmentationClass and SegmentationObject. You can use multiple datasets to train.


## Evaluation

```bash
python eval_ssd.py --net mb1-ssd  --dataset ~/data/VOC0712/test/VOC2007/ --trained_model models/mobilenet-v1-ssd-mp-0_675.pth --label_file models/voc-model-labels.txt 
```

## Convert models to ONNX and Caffe2 models

```bash
python convert_to_caffe2_models.py mb1-ssd models/mobilenet-v1-ssd-mp-0_675.pth models/voc-model-labels.txt 
```

The converted models are models/mobilenet-v1-ssd.onnx, models/mobilenet-v1-ssd_init_net.pb and models/mobilenet-v1-ssd_predict_net.pb. The models in the format of pbtxt are also saved for reference.

## Retrain on Open Images Dataset

Let's we are building a model to detect guns for security purpose.

Before you start you can try the demo.

```bash
wget -P models https://storage.googleapis.com/models-hao/gun_model_2.21.pth
wget -P models https://storage.googleapis.com/models-hao/open-images-model-labels.txt
python run_ssd_example.py mb1-ssd models/gun_model_2.21.pth models/open-images-model-labels.txt ~/Downloads/big.JPG
```

![image](https://user-images.githubusercontent.com/35050296/110277663-7b099b00-7fb4-11eb-8b7c-920b35515d58.png)


If you manage to get more annotated data, the accuracy could become much higher.

### Download data

```bash
python open_images_downloader.py --root ~/data/open_images --class_names "Handgun,Shotgun" --num_workers 20
```

It will download data into the folder ~/data/open_images.

The content of the data directory looks as follows.

```
class-descriptions-boxable.csv       test                        validation
sub-test-annotations-bbox.csv        test-annotations-bbox.csv   validation-annotations-bbox.csv
sub-train-annotations-bbox.csv       train
sub-validation-annotations-bbox.csv  train-annotations-bbox.csv
```

The folders train, test, validation contain the images. The files like sub-train-annotations-bbox.csv 
is the annotation file.

### Retrain

```bash
python train_ssd.py --dataset_type open_images --datasets ~/data/open_images --net mb1-ssd --pretrained_ssd models/mobilenet-v1-ssd-mp-0_675.pth --scheduler cosine --lr 0.01 --t_max 100 --validation_epochs 5 --num_epochs 100 --base_net_lr 0.001  --batch_size 5
```

You can freeze the base net, or all the layers except the prediction heads. 

```
  --freeze_base_net     Freeze base net layers.
  --freeze_net          Freeze all the layers except the prediction head.
```

You can also use different learning rates 
for the base net, the extra layers and the prediction heads.

```
  --lr LR, --learning-rate LR
  --base_net_lr BASE_NET_LR
                        initial learning rate for base net.
  --extra_layers_lr EXTRA_LAYERS_LR
```

As subsets of open images data can be very unbalanced, it also provides
a handy option to roughly balance the data.

```
  --balance_data        Balance training data by down-sampling more frequent
                        labels.
```

### Test on image

```bash
python run_ssd_example.py mb1-ssd models/mobilenet-v1-ssd-Epoch-99-Loss-2.2184619531035423.pth models/open-images-model-labels.txt ~/Downloads/gun.JPG
```


## ONNX Friendly VGG16 SSD

! The model is not really ONNX-Friendly due the issue mentioned here "https://github.com/qfgaohao/pytorch-ssd/issues/33#issuecomment-467533485"

The Scaled L2 Norm Layer has been replaced with BatchNorm to make the net ONNX compatible.

### Train

The pretrained based is borrowed from https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth .

```bash
python train_ssd.py --datasets ~/data/VOC0712/VOC2007/ ~/data/VOC0712/VOC2012/ --validation_dataset ~/data/VOC0712/test/VOC2007/ --net "vgg16-ssd" --base_net models/vgg16_reducedfc.pth  --batch_size 24 --num_epochs 150 --scheduler cosine --lr 0.0012 --t_max 150 --validation_epochs 5
```

### Eval

```bash
python eval_ssd.py --net vgg16-ssd  --dataset ~/data/VOC0712/test/VOC2007/ --trained_model models/vgg16-ssd-Epoch-115-Loss-2.819455094383535.pth --label_file models/voc-model-labels.txt
```

## TODO

1. Resnet34 Based Model.
2. BatchNorm Fusion.
