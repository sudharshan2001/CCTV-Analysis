# CCTV-Analytics

This application aims to enhance security and surveillance by leveraging deep learning techniques to automatically analyze and classify actions in CCTV footage, providing valuable insights and aiding in the identification of potential security threats.  This was presented at CyberX hackathon in final round.

### Data preparation
For training and evaluation, I have used the UCF101 crime dataset, which contains various criminal activities, as well as a custom dataset specific to the target surveillance scenario. The model has been trained on these datasets to learn and recognize different types of actions, enabling real-time monitoring and detection of suspicious activities in the CCTV camera feed.

### Model Architecture
The application utilizes the ResNet101 model to extract image embeddings, which are representations of images in a 2048-dimensional space. These embeddings capture important features of the actions being performed in the footage. Initiallly i explored 3D CNN but inorder to train faster in limited span of time and to reduce the inference period i went with this approach.

To classify the actions, I have incorporated LSTM (Long Short-Term Memory) networks. By leveraging the temporal information captured by LSTM, the application can accurately classify the actions observed in the CCTV footage.

# Installation

Clone this repository

```sh
git clone https://github.com/sudharshan2001/CCTV-Analytics.git
cd CCTV-Analytics
```

Create Conda Environment

```sh
conda create --name cctvanalysis python=3.9.7
conda  activate zfish
```

Install Requirements

```sh
pip install -r requirements.txt
```

>check this repository and download the models from it https://github.com/xinntao/ESRGAN.git
>Put the model inside req_packages/ESRGAN/models/ folder


```sh
python main.py
```

# Sample Result

![RA](https://github.com/sudharshan2001/CCTV-Analytics/assets/72936645/68a3c9f7-00d5-4387-b7f0-a6fb80fa868a)
