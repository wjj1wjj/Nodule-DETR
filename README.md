# Nodule-DETR
Nodule-DETR: A Novel DETR Architecture with Frequency-Channel Attention for Ultrasound Thyroid Nodule Detection  

This is the code repository for Nodule-DETR.   

# ✏️ Introduction  
Our network is structured as follows. For more details, please read the paper.  
<img width="3920" height="2470" alt="整体框图0628" src="https://github.com/user-attachments/assets/b27e94f8-ab38-42ad-ba55-26c04cdbba20" />

The MSFCA module is shown below.   
<img width="2792" height="1387" alt="fcanet0628" src="https://github.com/user-attachments/assets/965fb911-af44-4d95-abee-274a2b19300b" />

The HFF module is shown below. 
<img width="3464" height="1140" alt="HFF0628" src="https://github.com/user-attachments/assets/ecd85242-8950-4600-a8fc-acd2a8dcf91c" />

The MSDA module is shown below.   
<img width="4081" height="2333" alt="可变形注意力0628" src="https://github.com/user-attachments/assets/88ddc42b-c7fc-45c7-a437-ed0f44707572" />

# 📥 Installation and Usage  
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
# 📦Run
```
python Nodule-DETR/main.py
```
# 🤝 Contributing  
Feel free to submit issues or pull requests to contribute to the project and improve it.
# 📜 License  
This project is licensed under the MIT License. For more details, please refer to the LICENSE file.











