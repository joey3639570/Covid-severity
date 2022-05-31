from flask import Flask, request, jsonify
import os,sys
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pprint
import json
import torch
import torch.nn.functional as F
import torchvision, torchvision.transforms
import skimage, skimage.filters
import torchxrayvision as xrv
import pickle


# flask app 的定義，如果是合併到前端的話可以不管
app = Flask(__name__)

# 禁止 matplotlib 在 flask 中的輸出
plt.switch_backend('Agg')

# 許重複加載動態鏈接庫
# Ref.: https://blog.csdn.net/bingjianIT/article/details/86182096
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# API 的儲存路徑，裡面有
source_path = os.path.dirname(os.path.realpath(__file__))

"""
├── __pycache__
│   └── app.cpython-38.pyc
├── api.py # API 本體
├── backend-requirements.txt # API 需要的模組
├── images # 儲存上每影像的路徑
│   ├── 1-s2.0-S0140673620303706-fx1_lrg.jpg
├── saliency_images # 儲存上每影像的路徑
│   ├── 1-s2.0-S0140673620303706-fx1_lrg.jpg
└── weights # 儲存模型權重的路徑
    ├── mlp_geo.pkl # geographic_extent
    └── mlp_opa.pkl # opacity
"""
# Model 的 Class
class PneumoniaSeverityNetStonyBrook(torch.nn.Module):
    """
    Trained on Stony Brook data https://doi.org/10.5281/zenodo.4633999
    Radiographic Assessment of Lung Opacity Score Dataset
    """
    def __init__(self):
        super(PneumoniaSeverityNetStonyBrook, self).__init__()
        self.basemodel = xrv.models.DenseNet(weights="all")
        self.basemodel.op_threshs = None
        self.net_geo = self.create_network(os.path.join(source_path, "weights/mlp_geo.pkl"))
        self.net_opa = self.create_network(os.path.join(source_path, "weights/mlp_opa.pkl"))        

    def create_network(self, path):
        coefs, intercepts = pickle.load(open(path,"br"))
        layers = []
        for i, (coef, intercept) in enumerate(zip(coefs, intercepts)):
            la = torch.nn.Linear(*coef.shape)
            la.weight.data = torch.from_numpy(coef).T.float()
            la.bias.data = torch.from_numpy(intercept).T.float()
            layers.append(la)
            if i < len(coefs)-1: # if not last layer
                layers.append(torch.nn.ReLU())
                
        return torch.nn.Sequential(*layers)
        
    def forward(self, x):
        
        ret = {}
        ret["feats"] = self.basemodel.features2(x)
        
        ret["geographic_extent"] = self.net_geo(ret["feats"])[0]
        ret["geographic_extent"] = torch.clamp(ret["geographic_extent"],0,8)
        
        ret["opacity"] = self.net_opa(ret["feats"])[0]
        ret["opacity"] = torch.clamp(ret["opacity"],0,8)
        
        return ret

# 模型 inference 的 pipeline，會 return 出來一個 dict output
# output 包括：影像、opacity、geographic_extent
def process(model, img_path, cuda=False):
    
    img = skimage.io.imread(img_path)
    img = xrv.datasets.normalize(img, 255)  

    # Check that images are 2D arrays
    if len(img.shape) > 2:
        img = img[:, :, 0]
    if len(img.shape) < 2:
        print("error, dimension lower than 2 for image")

    # Add color channel
    img = img[None, :, :]                    
    
    transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),
                                                xrv.datasets.XRayResizer(224)])

    img = transform(img)
    
    with torch.no_grad():
        img = torch.from_numpy(img).unsqueeze(0)
        if cuda:
            img = img.cuda()
            model = model.cuda()

        outputs = model(img)
    
    outputs["img"] = img
    return outputs
    



    return jsonify(source_path)

# api 本體
@app.route("/api/v1/predict-by-image/", methods=["GET"])
def predict_by_image():

    # 從 Browser 裡面拿到的參數
    # e.g., http://host:port/api/v1/predict-by-image/?img_path=xxx
    img_name = request.args.get('img_name', default='', type=str)

    # 設定影像的路徑
    img_path = source_path + "/images/" + img_name
    saliency_path = source_path + "/saliency_images/" + img_name

    # 定義模型
    model = PneumoniaSeverityNetStonyBrook()

    # 執行 inference
    outputs = process(model, img_path)

    # 定義 opacity 和 geographic_extent 的資料格式
    geo = float(outputs["geographic_extent"].cpu().numpy())
    opa = float(outputs["opacity"].cpu().numpy())

    # 定義影像
    img = outputs["img"]
    img = img.requires_grad_()
    outputs = model(img)
    grads = torch.autograd.grad(outputs["geographic_extent"], img)[0][0][0]
    blurred = skimage.filters.gaussian(grads.detach().cpu().numpy()**2, sigma=(5, 5), truncate=3.5)

    my_dpi = 100
    fig = plt.figure(frameon=False, figsize=(224/my_dpi, 224/my_dpi), dpi=my_dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(img[0][0].detach().cpu().numpy(), cmap="gray", aspect='auto')
    ax.imshow(blurred, alpha=0.5)
    plt.ioff()
    plt.savefig(saliency_path)

    # 將模型輸出的資料轉成 json
    result = {"img_name": img_name,"img_path": img_path, "geographic_extent": geo, "opacity": opa,"saliency_path": saliency_path}

    return jsonify(result)

# Main function
if __name__ == '__main__':
    app.run(debug=True)

