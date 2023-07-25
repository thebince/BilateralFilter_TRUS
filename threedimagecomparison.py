import medpy.metric as md
import medpy.io as io
import medpy.features as mft
from medpy.io import save
import medpy.filter as mfil
from matplotlib import pyplot as plt
import numpy as np
from skimage import transform
from skimage.util import img_as_float
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import gaussian_gradient_magnitude
import numpy

#Load GIPL file
# image_us, image_header = io.load('/home/alanb/cwdata/test_trus.gipl')
# filtered_image, image_header = io.load('/home/alanb/cwdata/bifil3d_alanbince.gipl')

# image_us = img_as_float(image_us)
# print(image_us.dtype)
# print(filtered_image.dtype)

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
    #Load filtered 3D image file
    bi_image3d, image_header = io.load('bifil3d_trus.gipl')
    original_us = img_as_float(image_us)

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
    save(gimage3d,'./gimage3dresult.gipl',image_header)
    save(oimage3d,'./oimage3dresult.gipl',image_header)


