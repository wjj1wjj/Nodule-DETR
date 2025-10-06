# Nodule-DETR
Nodule-DETR: A Novel DETR Architecture with Frequency-Channel Attention for Ultrasound Thyroid Nodule Detection

# 📥 Installation and Usage
## 📦 Dependencies
1.📚Clone this repo  
```
git clone https://github.com/wjj1wjj/Nodule-DETR.git  
cd Nodule-DETR
```
2.🧩Install Pytorch and torchvision  
```
# an example:
conda create -n Nodule-DETR python=3.8
conda activate Nodule-DETR
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1  pytorch-cuda=11.8 -c pytorch -c nvidia
```
3.🛠️Install other needed packages
```
pip install -r requirements.txt
```
# 📋Data
Please organize your dataset as following: 
```
COCODIR/
  ├── train2017/
  ├── val2017/
  └── annotations/
  	├── instances_train2017.json
  	└── instances_val2017.json
```
# 📜Run
```
python Nodule-DETR/main.py
```









