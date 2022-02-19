In this Implementation of the CV task of Image SuperResolution, I’ve used  standard CNNs as well as CNNs containing a Residual Block.
In order to compare the different interpolation techniques used while upsampling our images, I’ve used the SRCNN to make direct comparisons between Bicubic, Bilinear and Nearest Neighbour interpolations.
Our input image is of the size 100x100, which is then upsampled using our prep_image function. 
In order to downscale the HR images from the dataset, the resize function has been used, along with a Gaussian Blur kernel, which serves to smoothen the noise generated when the image is downscaled.

