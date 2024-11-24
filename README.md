# NeoPolyp Segmentation


## About
This package contains my submission to [BKAI-IGH NeoPolyp competition](https://www.kaggle.com/c/bkai-igh-neopolyp/overview) on Kaggle.


## Usage

To infer an image, run the following commands:

1. Clone the repository:\
   ```git clone https://github.com/Decent-Cypher/NeoPolyp_Segmentation.git```
   
2. Navigate to the project directory:\
   ```cd NeoPolyp_Segmentation```

3. Install the necessary dependencies:\
   ```pip install -r requirements.txt```

4. Download the state dictionary of the pretrained model in this link, then put it inside a folder named "model":\
[model here](https://husteduvn-my.sharepoint.com/:u:/g/personal/binh_nd225475_sis_hust_edu_vn/EQw4URb9dLpMgYACDCpWTFgB-vSRcEGF6v7-YKjZ6PMEFg?e=WvVYV9)
   
3. Infer an image (put your image in this directory first, or use the sample_input file I provided):\
   ```python3 infer.py --image_path <image-name>.jpeg```
   
4. You can view your output image named output.jpg in the same directory
