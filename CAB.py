from keras.layers.core import Lambda
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.inception_v3 import InceptionV3 
from keras.applications.densenet import DenseNet121
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception
from keras.applications.mobilenet import MobileNet
from keras.layers import *
from keras.models import Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import *
from keras.utils import multi_gpu_model
from matplotlib import pyplot as plt
from keras.models import load_model
import keras.backend as K
import os

def Global_attention_block(inputs):
    shape=K.int_shape(inputs)
    x=AveragePooling2D(pool_size=(shape[1],shape[2])) (inputs)
    x=Conv2D(shape[3],1, padding='same') (x)
    x=Activation('relu') (x)
    x=Conv2D(shape[3],1, padding='same') (x)
    x=Activation('sigmoid') (x)
    C_A=Multiply()([x,inputs])
    
    x=Lambda(lambda x: K.mean(x,axis=-1,keepdims=True))  (C_A)
    x=Activation('sigmoid') (x)
    S_A=Multiply()([x,C_A])
    return S_A
    
    
def Category_attention_block(inputs,classes,k):
    shape=K.int_shape(inputs)
    F=Conv2D(k*classes,1, padding='same') (inputs)
    F=BatchNormalization() (F)
    F1=Activation('relu') (F)
    
    F2=F1
    x=GlobalMaxPool2D()(F2)
    
    x=Reshape((classes,k)) (x)
    S=Lambda(lambda x: K.mean(x,axis=-1,keepdims=False))  (x)
    
    x=Reshape((shape[1],shape[2],classes,k)) (F1)
    x=Lambda(lambda x: K.mean(x,axis=-1,keepdims=False))  (x)
    x=Multiply()([S,x])
    M=Lambda(lambda x: K.mean(x,axis=-1,keepdims=True))  (x)
    
    semantic=Multiply()([inputs,M])
    return semantic


import torch
from keras import backend as K
from CAB import Category_attention_block

if __name__ == '__main__':
    classes = 64
    k = 8
    block = Category_attention_block  # 实例化模块
    input_tensor = torch.rand(3, classes, 85, 85)  # 输入 N C H W
    output = block(input_tensor, classes, k)  # 调用模块

    print(input_tensor.size(), output.size())  # 打印输入和输出的形状