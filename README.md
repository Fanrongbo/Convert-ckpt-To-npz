# Convert-ckpt-To-npz
Convert the model weights from .ckpt file to .npz file.
If you train a classical model (such as Vgg16) using your own dataset by tensorflow, the weights of model you saved in is .ckpt file. But sometime you need to use .npz file to load in the model.
This time you can use our projection to complete the conversion from .ckpt to .npz file.
In this example, I used the original weights of Vgg16 and fineturning weights trained by myself (only contained the weights of FC layers).

1. load the .ckpt file trained by yourself;
2. load the .npz file downloaded by network;
3. read the keys name of original .npz file;
4. use np.savez to save the weights in a new .npz file;
5.Vertify and compare whether the name and shape of the dict from the new generated ,pyz file is correct.
