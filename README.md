# Detecção de Potenciais Focos de Reprodução de Mosquitos | YoloV4 - Darknet 

Este repositório representa a pesquisa de detecção de objetos aplicada ao problema de identificação de potenciais focos de reprodução de mosquitos do gênero Aedes, cujo o objetivo é delimitar e classificar as possíveis regiões de interesse em imagens ou vídeos. A rede neural utilizada como base para a transferência de aprendizado foi a [YOLOV4](https://github.com/roboflow-ai/darknet.git), que utiliza o [framework Darknet](https://pjreddie.com/darknet/yolo/).


<!--![gif](https://github.com/PedroFilhoEng/smart-city-canaa/blob/4511667a76d649b4ff5ea0c9815ace2bfc1179a1/Tutorial/gifs/GIF_inferencias_canaa_dos_carajas_yolov4.gif)-->
![gif](https://media.giphy.com/media/NMp90gcoa18Xgk43ht/giphy.gif)
![gif](https://media.giphy.com/media/QltfI6jrrzWOPhv9Cd/giphy.gif)

# Conjunto de Treinamento - Dataset
O [conjunto de dados (DATASET)](https://app.roboflow.com/ds/718N6C8kGj?key=6wbmJBk15G) utilizado no treinamento da rede, foi resultado da pesquisa de diferentes fontes de imagens e vídeos relacionados. O dataset foi anotado manualmente e possui duas classes: **Água** e **Lixo**. Para anotar os dados foi utilizada a plataforma [Roboflow](https://app.roboflow.com/). 
** O modelo e o dataset estão em desenvolvimento ativo e estão sujeitos a modificações. 

### Detalhes do Conjunto de Dados
O dataset totaliza 1729 imagens e 3179 anotações, dívididas em 1961 anotações para a classe Água e 1218 anotações para a classe Lixo.
#### Resumo:
- **Imagens**: 1729;
- **Anotações**: 3179;
- **Balanço de Classe**: Água:1961 | Lixo: 1218.
 
<img src="https://github.com/PedroFilhoEng/smart-city-canaa/blob/7f588b73b190ae22ae2ede5b7783a5062f3a3afa/test_batch2_pred.jpg" width="900">

<!--
** GPU Speed measures end-to-end time per image averaged over 5000 COCO val2017 images using a V100 GPU with batch size 32, and includes image preprocessing, PyTorch FP16 inference, postprocessing and NMS. EfficientDet data from [google/automl](https://github.com/google/automl) at batch size 8.

- **August 13, 2020**: [v3.0 release](https://github.com/ultralytics/yolov5/releases/tag/v3.0): nn.Hardswish() activations, data autodownload, native AMP.
- **July 23, 2020**: [v2.0 release](https://github.com/ultralytics/yolov5/releases/tag/v2.0): improved model definition, training and mAP.
- **June 22, 2020**: [PANet](https://arxiv.org/abs/1803.01534) updates: new heads, reduced parameters, improved speed and mAP [364fcfd](https://github.com/ultralytics/yolov5/commit/364fcfd7dba53f46edd4f04c037a039c0a287972).
- **June 19, 2020**: [FP16](https://pytorch.org/docs/stable/nn.html#torch.nn.Module.half) as new default for smaller checkpoints and faster inference [d4c6674](https://github.com/ultralytics/yolov5/commit/d4c6674c98e19df4c40e33a777610a18d1961145).
- **June 9, 2020**: [CSP](https://github.com/WongKinYiu/CrossStagePartialNetworks) updates: improved speed, size, and accuracy (credit to @WongKinYiu for CSP).
- **May 27, 2020**: Public release. YOLOv5 models are SOTA among all known YOLO implementations.


##Checkpoints Pré-Treinados

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
** Test Time Augmentation ([TTA](https://github.com/ultralytics/yolov5/issues/303)) runs at 3 image sizes. **Reproduce TTA** by `python test.py --data coco.yaml --img 832 --iou 0.65 --augment` -->

# CUDA
Para utilizar o comando nvcc do CUDA toolkit, execute:
```bash
!/usr/local/cuda/bin/nvcc --version
```
```bash
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2020 NVIDIA Corporation
Built on Wed_Jul_22_19:09:09_PDT_2020
Cuda compilation tools, release 11.0, V11.0.221
Build cuda_11.0_bu.TC445_37.28845127_0
Mon Mar 29 05:37:30 2021  
```
Para verificar a versão CUDA, execute:
```bash
!nvidia-smi
```
```bash
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 460.56       Driver Version: 460.32.03    CUDA Version: 11.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla T4            Off  | 00000000:00:04.0 Off |                    0 |
| N/A   59C    P8    11W /  70W |      0MiB / 15109MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

<!--
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
-->

## Ambiente

A rede neural costomizada pode ser executado no ambiente do Google Colab (com todas as dependências, incluindo [CUDA](https://developer.nvidia.com/cuda)/[CUDNN](https://developer.nvidia.com/cudnn), [Python](https://www.python.org/) and [PyTorch](https://pytorch.org/) pré-instalados):

- **Google Colab Notebook** com GPU grátis: <a href="https://colab.research.google.com/drive/1z12J8_8MeQYHNuFpScIgkV5cW-kQq4b9?authuser=6#scrollTo=W-GbIlPevNHR"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
<!-- - **Kaggle Notebook** with free GPU: [https://www.kaggle.com/ultralytics/yolov5](https://www.kaggle.com/ultralytics/yolov5)
- **Google Cloud** Deep Learning VM. See [GCP Quickstart Guide](https://github.com/ultralytics/yolov5/wiki/GCP-Quickstart) 
- **Docker Image** https://hub.docker.com/r/ultralytics/yolov5. See [Docker Quickstart Guide](https://github.com/ultralytics/yolov5/wiki/Docker-Quickstart) ![Docker Pulls](https://img.shields.io/docker/pulls/ultralytics/yolov5?logo=docker)-->


# Inferêcias

O script **detector.py** executa inferências em uma variedade de fontes, o exemplo ilustra a detecção em imagens e em vídeo.
```bash
%cd /content/darknet
!./darknet detector demo data/obj.data cfg/custom-yolov4-detector.cfg backup/yolov4.weights -dont_show ./video.mp4 -i 0 -out_filename ./inferencia_yolov4.mp4 # video
!./darknet detector test data/obj.data cfg/custom-yolov4-detector.cfg backup/yolov4.weights file.jpg  # image                            
```

**Para executar inferência em imagens de exemplo em `/canaa_dos_carajas/imagens`:**
```bash
$ !python detect.py --weights /content/canaa_dos_carajas/runs/train/yolov5s_results/weights/last.pt --img 416 --conf 0.4 --source /content/canaa_dos_carajas/imagens

CUDA-version: 11000 (11020), cuDNN: 7.6.5, GPU count: 1  
 OpenCV version: 3.2.0
 compute_capability = 700, cudnn_half = 0 
net.optimized_memory = 0 
mini_batch = 1, batch = 24, time_steps = 1, train = 0 
   layer   filters  size/strd(dil)      input                output
   0 conv     32       3 x 3/ 1    416 x 416 x   3 ->  416 x 416 x  32 0.299 BF
   1 conv     64       3 x 3/ 2    416 x 416 x  32 ->  208 x 208 x  64 1.595 BF
   2 conv     64       1 x 1/ 1    208 x 208 x  64 ->  208 x 208 x  64 0.354 BF
   3 route  1 		                           ->  208 x 208 x  64 
                                   .
                                   .
                                   .
 152 conv    512       3 x 3/ 2     26 x  26 x 256 ->   13 x  13 x 512 0.399 BF
 153 route  152 116 	                           ->   13 x  13 x1024 
 154 conv    512       1 x 1/ 1     13 x  13 x1024 ->   13 x  13 x 512 0.177 BF
 155 conv   1024       3 x 3/ 1     13 x  13 x 512 ->   13 x  13 x1024 1.595 BF
 156 conv    512       1 x 1/ 1     13 x  13 x1024 ->   13 x  13 x 512 0.177 BF
 157 conv   1024       3 x 3/ 1     13 x  13 x 512 ->   13 x  13 x1024 1.595 BF
 158 conv    512       1 x 1/ 1     13 x  13 x1024 ->   13 x  13 x 512 0.177 BF
 159 conv   1024       3 x 3/ 1     13 x  13 x 512 ->   13 x  13 x1024 1.595 BF
 160 conv     24       1 x 1/ 1     13 x  13 x1024 ->   13 x  13 x  24 0.008 BF
 161 yolo
[yolo] params: iou loss: ciou (4), iou_norm: 0.07, cls_norm: 1.00, scale_x_y: 1.05
nms_kind: greedynms (1), beta = 0.600000 
Total BFLOPS 59.578 
avg_outputs = 490041 
 Allocate additional workspace_size = 52.43 MB 
Loading weights from backup/custom-yolov4-detector_final.weights...
 seen 64, trained: 96 K-images (1 Kilo-batches_64) 
Done! Loaded 162 layers from weights-file 
test/1.jpg: Predicted in 11.755000 milli-seconds.
lixo: 85%
lixo: 78%
lixo: 83%
agua: 90%
Unable to init server: Could not connect: Connection refused

(predictions:4089): Gtk-WARNING **: 23:24:49.356: open display: 
```
<img src="https://github.com/PedroFilhoEng/smart-city-canaa/blob/fbb601c689c8f44612cac49348d364e1d590b0fc/Tutorial/gifs/resultado_2.jpg" width="480">

**Para executar inferência em vídeos em `/canaa_dos_carajas`:**
```bash
$ !python detect.py --weights /content/canaa_dos_carajas/runs/train/yolov5s_results/weights/last.pt --conf 0.4 --source video.mp4

Saving TESTE.mp4 to TESTE.mp4
User uploaded file "TESTE.mp4" with length 9710022 bytes
     |████████████████████████████████| 645kB 4.3MB/s 
Namespace(agnostic_nms=False, augment=False, classes=None, conf_thres=0.4, device='', exist_ok=False, img_size=640, iou_thres=0.45, name='exp', project='runs/detect', save_conf=False, save_txt=False, source='video.mp4', update=False, view_img=False, weights=['/content/canaa_dos_carajas/runs/train/yolov5s_results/weights/last.pt'])
YOLOv5 🚀 d8c50c2 torch 1.8.0+cu101 CPU

Fusing layers... 
Model Summary: 232 layers, 7249215 parameters, 0 gradients, 16.8 GFLOPS
video 1/1 (1/1800) /content/canaa_dos_carajas/video.mp4: 384x640 1 lixo, Done. (0.377s)
video 1/1 (2/1800) /content/canaa_dos_carajas/video.mp4: 384x640 1 lixo, Done. (0.281s)
video 1/1 (3/1800) /content/canaa_dos_carajas/video.mp4: 384x640 Done. (0.280s)
video 1/1 (4/1800) /content/canaa_dos_carajas/video.mp4: 384x640 Done. (0.276s)
video 1/1 (5/1800) /content/canaa_dos_carajas/video.mp4: 384x640 Done. (0.280s)
video 1/1 (6/1800) /content/canaa_dos_carajas/video.mp4: 384x640 Done. (0.278s)
video 1/1 (7/1800) /content/canaa_dos_carajas/video.mp4: 384x640 Done. (0.282s)
video 1/1 (8/1800) /content/canaa_dos_carajas/video.mp4: 384x640 Done. (0.272s)
video 1/1 (9/1800) /content/canaa_dos_carajas/video.mp4: 384x640 Done. (0.276s)
video 1/1 (10/1800) /content/canaa_dos_carajas/video.mp4: 384x640 Done. (0.282s)
video 1/1 (11/1800) /content/canaa_dos_carajas/video.mp4: 384x640 Done. (0.284s)
video 1/1 (12/1800) /content/canaa_dos_carajas/video.mp4: 384x640 Done. (0.282s)
                                 .
                                 .
                                 .
video 1/1 (1797/1800) /content/canaa_dos_carajas/video.mp4: 384x640 1 lixo, Done. (0.276s)
video 1/1 (1798/1800) /content/canaa_dos_carajas/video.mp4: 384x640 1 lixo, Done. (0.280s)
video 1/1 (1799/1800) /content/canaa_dos_carajas/video.mp4: 384x640 1 lixo, Done. (0.281s)
video 1/1 (1800/1800) /content/canaa_dos_carajas/video.mp4: 384x640 Done. (0.278s)
Results saved to runs/detect/exp
Done. (535.095s)
```

![gif](https://github.com/PedroFilhoEng/smart-city-canaa/blob/dfe41ba4e5340ac5c8aee2cf3498160a99fdc407/Animated%20GIF-downsized_large.gif)


# Treino

Faça o download de [COCO](https://github.com/ultralytics/yolov5/blob/master/data/scripts/get_coco.sh) e execute o comando abaixo. Tempos de treinamento para YOLOv5s/m/l/x são em média 2/4/6/8 dias em um único V100 (multi-GPU times faster). Usar o maior `--batch-size` exige uma maior GPU, caso a GPU não tenha alta capacidade, optar por --batch-size 16.
Este tutorial utiliza a arquitetura YOLOv5s.  

```bash
$ python train.py --data coco.yaml --cfg yolov5s.yaml --weights '' --batch-size 64
                                         yolov5m                                40
                                         yolov5l                                24
                                         yolov5x                                16
```
Neste tutorial o comando para treino foi o seguinte:
```bash
$ !python train.py --data '../data.yaml' --cfg ./models/custom_yolov5s.yaml --weights ''--batch 16  --img 416  --epochs 4000  --name yolov5s_results  --cache
```
Argumentos:
- **img:** define o tamanho da imagem de entrada
- **batch:** determina o batch
- **epochs:** define o número de épocas de treinamento. (Obs: é comum definir epochs = 2000*número_de_classes)
- **data:** define o caminho para o arquivo yaml
- **cfg:** especifica a configuração do modelo
- **weights:** especifica um caminho personalizado para os pesos. (Obs.: você pode baixar os pesos da [Pasta](https://github.com/PedroFilhoEng/canaa_dos_carajas/blob/d8c50c2810130aa0722f92791fbbba46a16f0944/runs/train/yolov5s_results/weights))
- **name:** nome dos resultados
- **nosave:** salva apenas na última época
- **cache:** armazenas as imagens em cache para agilizar o treino
<img src="https://github.com/PedroFilhoEng/smart-city-canaa/blob/f9070052d45053c15af8b98c92c667cfb71eeef2/Tutorial/resultados_yolov4.png" width="600">
