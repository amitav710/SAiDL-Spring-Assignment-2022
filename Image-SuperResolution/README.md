In this Implementation of the CV task of Image SuperResolution, I’ve used  standard CNNs as well as CNNs containing a Residual Block.
In order to compare the different interpolation techniques used while upsampling our images, I’ve used the SRCNN to make direct comparisons between Bicubic, Bilinear and Nearest Neighbour interpolations.
Our input image is of the size 100x100, which is then upsampled using our prep_image function. 
In order to downscale the HR images from the dataset, the resize function has been used, along with a Gaussian Blur kernel, which serves to smoothen the noise generated when the image is downscaled.

![image](https://user-images.githubusercontent.com/77532573/154813490-746add38-43b4-42ae-b625-389499d14836.png)
![image](https://user-images.githubusercontent.com/77532573/154813507-5ed172c0-4c36-4349-a5f1-b56d7d4a0753.png)

Here we can see our use of the different methods of interpolation and our prep_image function, which helps us generate the 300x300 inputs for our model.

The model used to compare these upscaling techniques was the same for all.

![image](https://user-images.githubusercontent.com/77532573/154813529-7b6d3217-658b-44d4-bf77-73df4356aa38.png)

We used the popularly used 9-3-5 structure, which is much more efficient than our 9-5-5 structure of kernels and the accuracy to efficiency trade-off makes it worth it to use it over the 9-1-5 structure.

![image](https://user-images.githubusercontent.com/77532573/154813541-40f2de37-ac56-41ed-aa42-ffb98f1d2328.png)

We use 128, 64 and 3 filters (no. of kernels) respectively in our three Convolutional Layers. This gave our model a total of 109,000 trainable parameters. Referring to literature available online, we have also opted to use the glorot_uniform kernel initializiers. Padding is kept as same to maintain the size of our input. We use ReLU as our activation for the first two layers and linear activation in our last layer, as we find it leads to quicker convergence. We use the Adam optimizer for our fit function, due to its versatility. After a little experimentation, it is found that setting the learning rate to 0.0003 gives optimum values for our loss after 25 epochs, which is the number of epochs we train our model for.
The loss function and metric for all the models I used were Mean Squared Error. The PSNR can also be used as our metric, but it is effectively the same thing to use MSE, as shown by its formula:
![image](https://user-images.githubusercontent.com/77532573/154813556-e1ec413e-087c-49ff-9bbf-d3b46f331e87.png)

Here MAXI is the maximum possible pixel value of the image

We then use our test dataset to validate our results. To my surprise, the Nearest Neighbour method of interpolation seemed to outperform the Bilinear interpolation on a majority of the images. However, all methods of interpolation seemed to improve upon the quality of the input images generated by the prep_image function. 
To compare these images, we look at them side by side on a particular image of a grey cat, taken from the test set.

![image](https://user-images.githubusercontent.com/77532573/154813577-55c0add6-32d6-45a4-a09c-cab9cbabd45d.png)

Ground Truth HR image from Test Set

![image](https://user-images.githubusercontent.com/77532573/154813611-374bb4b3-9b6c-4d7c-bfeb-75434717d564.png)

 Input image after Bicubic Interpolation: PSNR=28.247568

![image](https://user-images.githubusercontent.com/77532573/154813623-c7f41655-3d37-4091-b4ee-aad62bc1c7a2.png)

  Output image of SRCNN_bicubic:PSNR=29.184723
  
![image](https://user-images.githubusercontent.com/77532573/154813630-0e7ca894-1198-408c-aac7-4ad143dd6ec9.png)

Input image after Bilinear Interpolation: PSNR= 27.729452

![image](https://user-images.githubusercontent.com/77532573/154813642-0ba08033-f773-4587-b678-ebb766335687.png)

Output image of SRCNN_bilinear:PSNR= 28.674414

![image](https://user-images.githubusercontent.com/77532573/154813673-8d83318f-fb6d-4b5a-b82d-982fdd29da6e.png)

Input image after Nearest Neighbour Interpolation:PSNR= 26.75208

![image](https://user-images.githubusercontent.com/77532573/154813682-7aa342c4-972e-4596-860b-7f3ba2dfe531.png)

Output image of SRCNN_NN:PSNR= 28.94901

Moving on to our ResNet models, at first I tried using two models using different depths and number of parameters. In these models, I have specifically tried to compare the sub-pixel and transposed convolutional methods of upscaling our images. None of these models have used inputs that have gone through any form of upsampling interpolations, hence I did not expect them to outperform the technique used in prep_image in my SRCNN models.
In the first Subpixel model, I have used the same low_res function to generate its inputs. This model uses kernels of sizes 5 and 3, has one residual block, containing a couple of BatchNorm and ReLU layers. In the end, the tensorflow function depth_to_space is used to generate r^2 channels, where r is our upsampling ratio.

![image](https://user-images.githubusercontent.com/77532573/154813701-8089c9f4-53d9-48f5-8a2e-c1fb6a469557.png)

Subpixel1

![image](https://user-images.githubusercontent.com/77532573/154813711-3c931cab-0295-4e1d-ab40-f89b87c98d04.png)

This model generated 328,000 trainable parameters and was trained for 75 epochs using the Adam optimizer with a learning rate of 0.0003 and then 0.0001, along with some callbacks.
A second model was also tested out which used the subpixel method for 75 epochs, keeping most things same except that they did not contain the BatchNorm layers in the residual block and it suffered heavily in terms of its loss, deeming BatchNormalization to be very effective.

![image](https://user-images.githubusercontent.com/77532573/154813723-57fc26d0-328b-4d42-a08d-bf59cb4e6ac8.png)

Subpixel2

The transposed convolutional layer of roughly the same no. of trainable parameters which used the Conv2DTranspose layer of keras, suffered heavily too, not giving nearly the same performance as our subpixel1 model.

![image](https://user-images.githubusercontent.com/77532573/154813741-a6c7059a-c37c-4c03-bb41-9a52a7f8a508.png)

Transposed_conv_end

We also opted to use the transpose function at the end of the model as otherwise it was proving to be very heavy for the gpu to process 300x300 inputs at every layers.
Using our same cat image to compare the results using PSNR as our metric, we can clearly see the difference in results:

![image](https://user-images.githubusercontent.com/77532573/154813767-1ad38218-50c9-4262-8b86-8580d2a1b060.png)

Subpixel1: PSNR=28.322552

![image](https://user-images.githubusercontent.com/77532573/154813779-6583d470-0d22-4e87-b148-e024f9ef1b98.png)

Subpixel2: PSNR=26.036636

![image](https://user-images.githubusercontent.com/77532573/154813787-4443b169-ef5d-4721-a3ae-410971833a65.png)

Transposed_conv_end:PSNR=26.066544

Given more time and GPU resources, I would work upon improving these model’s performances, explore the bonus tasks such as the Cutout augmentation as well as explore the architecture and working of the state-of-the-art SRGANs.






