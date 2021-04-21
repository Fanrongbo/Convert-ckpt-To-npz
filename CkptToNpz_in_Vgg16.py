import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

checkpoint_path="./model/model.ckpt-19999"#your ckpt path
reader=pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map=reader.get_variable_to_shape_map()

fine_turning={}
name=['fc6_b','fc6_W','fc7_b','fc7_W','fc8_b','fc8_W']
fine_turning={}
for id in name:
    fine_turning[id]=[]#generate a empty dict 
num=0
#Read the weights in the CKPT file in order
for key in var_to_shape_map:
    str_name = key
    # Because the model is optimized by ADAM algorithm, there is 'Adam' and 'beta' suffix Tensor in the generated CKPT
    if str_name.find('Adam') > -1:
        continue
    if str_name.find('beta') > -1:
        continue
    print("variable name and shape: ", key,reader.get_tensor(key).shape)
    fine_turning[name[num]]=reader.get_tensor(key)
    print(key,': complete!',name[num])
    num=num+1
# save npy
data_dict = np.load('vgg16_weights.npz')#original pre-training weights downloaded by network
keys: ['conv1_1_W', 'conv1_1_b', 'conv1_2_W', 'conv1_2_b', 'conv2_1_W', 'conv2_1_b', 'conv2_2_W', 'conv2_2_b', 'conv3_1_W', 'conv3_1_b', 'conv3_2_W', 'conv3_2_b', 'conv3_3_W', 'conv3_3_b', 'conv4_1_W', 'conv4_1_b', 'conv4_2_W', 'conv4_2_b', 'conv4_3_W', 'conv4_3_b', 'conv5_1_W', 'conv5_1_b', 'conv5_2_W', 'conv5_2_b', 'conv5_3_W', 'conv5_3_b', 'fc6_W', 'fc6_b', 'fc7_W', 'fc7_b', 'fc8_W', 'fc8_b']#the keys name of data_dict
c=np.savez('vgg_16_new.npz',conv1_1_W=data_dict['conv1_1_W'], 
                              conv1_1_b=data_dict['conv1_1_b'], 
                              conv1_2_W=data_dict['conv1_2_W'], 
                              conv1_2_b=data_dict['conv1_2_b'], 
                              conv2_1_W=data_dict['conv2_1_W'], 
                              conv2_1_b=data_dict['conv2_1_b'], 
                              conv2_2_W=data_dict['conv2_2_W'], 
                              conv2_2_b=data_dict['conv2_2_b'], 
                              conv3_1_W=data_dict['conv3_1_W'], 
                              conv3_1_b=data_dict['conv3_1_b'], 
                              conv3_2_W=data_dict['conv3_2_W'], 
                              conv3_2_b=data_dict['conv3_2_b'], 
                              conv3_3_W=data_dict['conv3_3_W'], 
                              conv3_3_b=data_dict['conv3_3_b'], 
                              conv4_1_W=data_dict['conv4_1_W'], 
                              conv4_1_b=data_dict['conv4_1_b'], 
                              conv4_2_W=data_dict['conv4_2_W'], 
                              conv4_2_b=data_dict['conv4_2_b'], 
                              conv4_3_W=data_dict['conv4_3_W'], 
                              conv4_3_b=data_dict['conv4_3_b'], 
                              conv5_1_W=data_dict['conv5_1_W'], 
                              conv5_1_b=data_dict['conv5_1_b'], 
                              conv5_2_W=data_dict['conv5_2_W'], 
                              conv5_2_b=data_dict['conv5_2_b'], 
                              conv5_3_W=data_dict['conv5_3_W'], 
                              conv5_3_b=data_dict['conv5_3_b'],
                              fc6_W=fine_turning['fc6_W'],
                              fc6_b=fine_turning['fc6_b'],
                              fc7_W=fine_turning['fc7_W'],
                              fc7_b=fine_turning['fc7_b'],
                              fc8_W=fine_turning['fc8_W'],
                              fc8_b=fine_turning['fc8_b'])
print('save npy over...')
print("-----------------------------------------")
print('Vertify and compare whether the name and shape of the dict from the new generated ,pyz file is correct!')
data_dict_new = np.load('vgg_16_new.npz', encoding='latin1')# 键值对的形式存在
keys_new = sorted(data_dict_new.keys())
print("keys_new:",keys_new)
print("-----------------------------------------")
data_dict_original = np.load('vgg16_weights.npz', encoding='latin1')# 键值对的形式存在
keys_original = sorted(data_dict_original.keys())
print("keys_original:",keys_original)
print("-----------------------------------------")
for key in keys_new:
    weights = data_dict_new[key]
    weights2 = data_dict_original[key]
    print()
    #compare the shape of each keys between the generated .npz file and original .npz file
    print(key,'weights shape: ', weights.shape,weights2.shape)
