# Detecção de Potenciais Focos de Reprodução de Mosquitos | YoloV4 - Darknet 

Este repositório representa a pesquisa de detecção de objetos aplicada ao problema de identificação de potenciais focos de reprodução de mosquitos do gênero Aedes, cujo o objetivo é delimitar e classificar as possíveis regiões de interesse em imagens ou vídeos. A rede neural utilizada como base para a transferência de aprendizado foi a [YOLOV4](https://github.com/roboflow-ai/darknet.git), que utiliza o [framework Darknet](https://pjreddie.com/darknet/yolo/).


![gif](https://github.com/PedroFilhoEng/smart-city-canaa/blob/4511667a76d649b4ff5ea0c9815ace2bfc1179a1/Tutorial/gifs/GIF_inferencias_canaa_dos_carajas_yolov4.gif)

# Conjunto de Treinamento - Dataset
O [conjunto de dados (DATASET)](https://app.roboflow.com/ds/718N6C8kGj?key=6wbmJBk15G) utilizado no treinamento da rede, foi resultado da pesquisa de diferentes fontes de imagens e vídeos relacionados. O dataset foi anotado manualmente e possui duas classes: **Água** e **Lixo**. Para anotar os dados foi utilizada a plataforma [Roboflow](https://app.roboflow.com/). 
** O modelo e o dataset estão em desenvolvimento ativo e estão sujeitos a modificações. 

### Detalhes do Conjunto de Dados
O dataset totaliza 1191 imagens e 1786 anotações, dívididas em 1156 anotações para a classe Água e 630 anotações para a classe Lixo.
#### Resumo:
- **Imagens**: 1191;
- **Anotações**: 1786;
- **Balanço de Classe**: Água:1156 | Lixo: 630.
 
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

# Requisitos

Python 3.8 ou superior com todas as dependências de [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) instaladas, incluindo `torch>=1.7`. Para instalar, execute:
```bash
$ pip install -r requirements.txt
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

- **Google Colab Notebook** com GPU grátis: <a href="https://colab.research.google.com/drive/1nuAv67diZcttBfKbg-DbcdPLw6yIREKk?authuser=3"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
<!-- - **Kaggle Notebook** with free GPU: [https://www.kaggle.com/ultralytics/yolov5](https://www.kaggle.com/ultralytics/yolov5)
- **Google Cloud** Deep Learning VM. See [GCP Quickstart Guide](https://github.com/ultralytics/yolov5/wiki/GCP-Quickstart) 
- **Docker Image** https://hub.docker.com/r/ultralytics/yolov5. See [Docker Quickstart Guide](https://github.com/ultralytics/yolov5/wiki/Docker-Quickstart) ![Docker Pulls](https://img.shields.io/docker/pulls/ultralytics/yolov5?logo=docker)-->


# Inferêcias

O script **detect.py** executa inferências em uma variedade de fontes, baixando modelos automaticamente da [versão mais recente do YOLOv5](https://github.com/ultralytics/yolov5/releases) e salvando os resultados em `runs/detect`.
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

**Para executar inferência em imagens de exemplo em `/canaa_dos_carajas/imagens`:**
```bash
$ !python detect.py --weights /content/canaa_dos_carajas/runs/train/yolov5s_results/weights/last.pt --img 416 --conf 0.4 --source /content/canaa_dos_carajas/imagens

Namespace(agnostic_nms=False, augment=False, classes=None, conf_thres=0.4, device='', exist_ok=False, img_size=416, iou_thres=0.45, name='exp', project='runs/detect', save_conf=False, save_txt=False, source='/content/canaa_dos_carajas/imagens', update=False, view_img=False, weights=['/content/canaa_dos_carajas/runs/train/yolov5s_results/weights/last.pt'])
YOLOv5 🚀 d8c50c2 torch 1.8.0+cu101 CPU

Fusing layers... 
Model Summary: 232 layers, 7249215 parameters, 0 gradients, 16.8 GFLOPS
image 1/4 /content/canaa_dos_carajas/imagens/caixa-dagua--2-_jpg.rf.2169d853ce03d7c936bfe85eda2320dd.jpg: 416x416 2 aguas, Done. (0.217s)
image 2/4 /content/canaa_dos_carajas/imagens/caixa-dagua--2-_jpg.rf.424082d0c3033a440af3508f468b6c2b.jpg: 416x416 2 aguas, Done. (0.215s)
image 3/4 /content/canaa_dos_carajas/imagens/caixa-dagua--2-_jpg.rf.eeee35b53575bd3d8926759848ed9a3b.jpg: 416x416 2 aguas, Done. (0.207s)
image 4/4 /content/canaa_dos_carajas/imagens/caixa-dagua--3-_jpg.rf.0d89f19be42dd7abcaa7a8687862cd03.jpg: 416x416 2 aguas, Done. (0.221s)
Results saved to runs/detect/exp2
Done. (0.912s)
```
<img src="https://user-images.githubusercontent.com/35050296/110505082-97085c00-80dc-11eb-8174-7b45270e4a28.png" width="480">

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










# Detecção de Potenciais Focos de Reprodução de Mosquitos | YoloV5 - Darknet 

Este repositório representa a pesquisa de detecção de objetos aplicada ao problema de identificação de potenciais focos de reprodução de mosquitos do gênero Aedes, cujo o objetivo é delimitar e classificar as possíveis regiões de interesse em imagens ou vídeos. A rede neural utilizada como base para a transferência de aprendizado foi a [YOLOV4](https://github.com/roboflow-ai/darknet.git), que utiliza o [framework Darknet](https://pjreddie.com/darknet/yolo/).



![gif](https://github.com/PedroFilhoEng/smart-city-canaa/blob/f9070052d45053c15af8b98c92c667cfb71eeef2/Tutorial/gifs/canna_fazenda_yoloV5.gif)

# Conjunto de Treinamento - Dataset
O [conjunto de dados (DATASET)](https://app.roboflow.com/ds/718N6C8kGj?key=6wbmJBk15G) utilizado no treinamento da rede, foi resultado da pesquisa de diferentes fontes de imagens e vídeos relacionados. O dataset foi anotado manualmente e possui duas classes: **Água** e **Lixo**. Para anotar os dados foi utilizada a plataforma [Roboflow](https://app.roboflow.com/). 
** O modelo e o dataset estão em desenvolvimento ativo e estão sujeitos a modificações. 

### Detalhes do Conjunto de Dados
O dataset totaliza 1191 imagens e 1786 anotações, dívididas em 1156 anotações para a classe Água e 630 anotações para a classe Lixo.
#### Resumo:
- **Imagens**: 1191;
- **Anotações**: 1786;
- **Balanço de Classe**: Água:1156 | Lixo: 630.
 
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

# Requisitos

Python 3.8 ou superior com todas as dependências de [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) instaladas, incluindo `torch>=1.7`. Para instalar, execute:
```bash
$ pip install -r requirements.txt
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

- **Google Colab Notebook** com GPU grátis: <a href="https://colab.research.google.com/drive/1Y3yoY1E_IEntfHdXeX3DmU_VUgvGTETg?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
<!-- - **Kaggle Notebook** with free GPU: [https://www.kaggle.com/ultralytics/yolov5](https://www.kaggle.com/ultralytics/yolov5)
- **Google Cloud** Deep Learning VM. See [GCP Quickstart Guide](https://github.com/ultralytics/yolov5/wiki/GCP-Quickstart) 
- **Docker Image** https://hub.docker.com/r/ultralytics/yolov5. See [Docker Quickstart Guide](https://github.com/ultralytics/yolov5/wiki/Docker-Quickstart) ![Docker Pulls](https://img.shields.io/docker/pulls/ultralytics/yolov5?logo=docker)-->


# Inferêcias

O script **detect.py** executa inferências em uma variedade de fontes, baixando modelos automaticamente da [versão mais recente do YOLOv5](https://github.com/ultralytics/yolov5/releases) e salvando os resultados em `runs/detect`.
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

**Para executar inferência em imagens de exemplo em `/canaa_dos_carajas/imagens`:**
```bash
$ !python detect.py --weights /content/canaa_dos_carajas/runs/train/yolov5s_results/weights/last.pt --img 416 --conf 0.4 --source /content/canaa_dos_carajas/imagens

Namespace(agnostic_nms=False, augment=False, classes=None, conf_thres=0.4, device='', exist_ok=False, img_size=416, iou_thres=0.45, name='exp', project='runs/detect', save_conf=False, save_txt=False, source='/content/canaa_dos_carajas/imagens', update=False, view_img=False, weights=['/content/canaa_dos_carajas/runs/train/yolov5s_results/weights/last.pt'])
YOLOv5 🚀 d8c50c2 torch 1.8.0+cu101 CPU

Fusing layers... 
Model Summary: 232 layers, 7249215 parameters, 0 gradients, 16.8 GFLOPS
image 1/4 /content/canaa_dos_carajas/imagens/caixa-dagua--2-_jpg.rf.2169d853ce03d7c936bfe85eda2320dd.jpg: 416x416 2 aguas, Done. (0.217s)
image 2/4 /content/canaa_dos_carajas/imagens/caixa-dagua--2-_jpg.rf.424082d0c3033a440af3508f468b6c2b.jpg: 416x416 2 aguas, Done. (0.215s)
image 3/4 /content/canaa_dos_carajas/imagens/caixa-dagua--2-_jpg.rf.eeee35b53575bd3d8926759848ed9a3b.jpg: 416x416 2 aguas, Done. (0.207s)
image 4/4 /content/canaa_dos_carajas/imagens/caixa-dagua--3-_jpg.rf.0d89f19be42dd7abcaa7a8687862cd03.jpg: 416x416 2 aguas, Done. (0.221s)
Results saved to runs/detect/exp2
Done. (0.912s)
```
<img src="https://user-images.githubusercontent.com/35050296/110505082-97085c00-80dc-11eb-8174-7b45270e4a28.png" width="480">

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
<img src="https://github.com/PedroFilhoEng/smart-city-canaa/blob/76c7e18c87f2e1f89f47c2daabbebcd5083963fe/results.png" width="900">








<!--
## Citation

[![DOI](https://zenodo.org/badge/264818686.svg)](https://zenodo.org/badge/latestdoi/264818686)
-->
<!--
## About Us

Ultralytics is a U.S.-based particle physics and AI startup with over 6 years of expertise supporting government, academic and business clients. We offer a wide range of vision AI services, spanning from simple expert advice up to delivery of fully customized, end-to-end production solutions, including:
- **Cloud-based AI** systems operating on **hundreds of HD video streams in realtime.**
- **Edge AI** integrated into custom iOS and Android apps for realtime **30 FPS video inference.**
- **Custom data training**, hyperparameter evolution, and model exportation to any destination.

For business inquiries and professional support requests please visit us at https://www.ultralytics.com. 
-->
<!--
## Contact

**Issues should be raised directly in the repository.** For business inquiries or professional support requests please visit https://www.ultralytics.com or email Glenn Jocher at glenn.jocher@ultralytics.com. 
-->
<!--
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
-->

