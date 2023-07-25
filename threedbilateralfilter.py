import medpy.io as io
from medpy.io import save
from matplotlib import pyplot as plt
import numpy as np
from skimage import transform
import medpy.metric as md
from skimage.metrics import structural_similarity as ssim
from skimage.util import img_as_float
from scipy.ndimage import gaussian_gradient_magnitude
import numpy
import time

#Load GIPL file
# image_us, image_header = io.load('/home/alanb/cwdata/test_trus.gipl')


#-------------------------3D BILATERAL FILTER-------------------------
def gaussianfilterfn(a,sigma):
    """
    Gaussian function for bilateral filter function. Remove high frequency component from the image.

    INPUTS:
    
    a: Value for plotting/calculating gaussian 
    sigma: standard deviation 

    OUTPUTS:

    k: Output from the Gaussian equation
    
    """

    #Gaussian equation
    k = 0
    k = (1.0/(2*numpy.pi*(sigma**2)))*numpy.exp(-(a**2)/(2*(sigma**2)))
    return k

#Euclidean Distance
def euclideandistance3d(x0,y0,z0,x1,y1,z1):
    """
    Euclidean distance function for 2D

    INPUTS:
    
    x0: Point value across x 
    y0: Point value across y
    z0: Point value across z
    x1: Point value across x
    y1: Point value across y
    z1: Point value across z

    OUTPUTS:

    edistance3d: Output of euclidean distance
    
    """
    #Euclidean equation
    edistance3d = 0
    edistance3d = numpy.sqrt(numpy.abs((x0-x1)**2+(y0-y1)**2+(z0-z1)**2))
    return edistance3d

#Define Bilateral Filter with two sigma values
def bilateral_filter3d(input_image, d, sigma_range3d, sigma_domain3d):
    """
    Bilateral filter for 3D image. Contains two sigma values.

    INPUTS:
    
    input_image: image volume - (type: Numpy array)
    d: Neighbourhood diameter, ideally leass than 7 - (datatype: int)
    sigma_range3d: Standard deviation of the range filter - (datatype: float)
    sigma_domain3d: Standard deviation of the range filter - (datatype: float)

    OUTPUTS:

    new_filter: filtered image output - (type:Numpy array)
    
    """

    start = time.time()
    #Pre-allocate values for the filtered image array
    new_filter = numpy.zeros(input_image.shape)

    #Iteration across three dimensions of the image
    for r in range(input_image.shape[0]):
        for c in range(input_image.shape[1]):
            for b in range(input_image.shape[2]):
                total_weight = 0
                filtered_image = 0
                #Iteration across neighbourhood diameter d
                for i in range(d):
                    for j in range(d):
                        for k in range(d):
                            #Difference across each point in row and column 
                            xx =r - (d/2 - i)
                            yy =c - (d/2 - j)
                            zz =b - (d/2 - k)
                            #New dimensions for computing gaussian kernel
                            if xx >= input_image.shape[0]:
                                xx = xx - 455
                            if yy >= input_image.shape[1]:
                                yy = yy - 325
                            if zz >= input_image.shape[2]:
                                zz = zz - 46
                            #Gaussian kernels for convolution
                            g_range3d = gaussianfilterfn(input_image[int(xx)][int(yy)][int(zz)] - input_image[r][c][b], sigma_range3d)
                            g_domain3d = gaussianfilterfn(euclideandistance3d(xx, yy, zz, r, c, b), sigma_domain3d)
                            # Convolution
                            wp = g_range3d * g_domain3d
                            filtered_image = (filtered_image) + (input_image[int(xx)][int(yy)][int(zz)] * wp)
                            total_weight = total_weight + wp
                filtered_image = filtered_image // total_weight
                new_filter[r][c][b] = int(numpy.round(filtered_image))

    stop = time.time()    
    print('Timer of the 3D Bilateral:',stop-start)        
    return new_filter

def MSE(A,B):
    """
    To calculate Mean Square Error(MSE)

    INPUTS:
    
    A: original 3D image - (type: Numpy array with float values)
    B: filtered 3D image - (type: Numpy array with float values)

    OUTPUTS:

    error: The output of MSE - (datatype:float)
    
    """
	# sum of the squared difference between the two images;
	# the two images must have the same dimension
    error = np.sum((A-B)**2)
    error = error/float(A.shape[0]*A.shape[1]*A.shape[2])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
    return error

def master():

    #Load 3D image file
    image_us, image_header = io.load('test_trus.gipl')
    original_us = img_as_float(image_us)

    #Image shape
    print(image_us.shape)

    #Voxel spacing
    print(image_header.get_voxel_spacing())
    
    #Running the 3D bilateral filter for 3D image volume
    kernel_size = 7
    print(kernel_size)
    #Bilateral filter function
    bi_image3d = bilateral_filter3d(input_image=image_us, d= kernel_size, sigma_range3d= 18 , sigma_domain3d=30)
    print(bi_image3d) 
    save(bi_image3d,'./bilateral3dresult.gipl',image_header)

    #MSE 
    meansquareerror = MSE(A=original_us,B=bi_image3d)
    #SSIM
    # print(image_us.dtype)
    # print(bi_image3d.dtype)
    structsim = ssim(im1=original_us,im2=bi_image3d)
    print('Mean Square Error of original and filtered 3D images:',meansquareerror)
    print('Structural Similarity Measure of original and filtered 3D images:',structsim)
    #Dice Coefficient
    dice_coeff = md.binary.dc(original_us,bi_image3d)
    print('Dice Coefficient:',dice_coeff)
    #Specificity
    spec = md.binary.specificity(original_us,bi_image3d)
    print('Specificity:',spec)
    #Sensitivity
    sens = md.binary.sensitivity(original_us,bi_image3d)
    print('Sensitivity:',sens)
    #Mutual Information
    mutual_info = md.image.mutual_information(original_us,bi_image3d)
    print('mutual information:',mutual_info)

    #Gaussian Gradient Magnitude
    gimage3d = gaussian_gradient_magnitude(bi_image3d,sigma=5)
    oimage3d = gaussian_gradient_magnitude(image_us,sigma=5)
    print('Gradient of Original:',oimage3d)
    print('Gradient of Filtered:',gimage3d)
    save(gimage3d,'./gimage3dres.gipl',image_header)
    save(oimage3d,'./oimage3dres.gipl',image_header)
    # Gaussian Gradient Magnitudes
    # save(gimage3d,'/home/alanb/cwdata/gimage3dresult.gipl',image_header)
    # save(oimage3d,'/home/alanb/cwdata/oimage3dresult.gipl',image_header)