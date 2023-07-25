import medpy.io as io
from medpy.io import save
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from skimage.transform import rotate
from skimage.metrics import structural_similarity as ssim
from skimage.util import img_as_float
import time
from skimage import feature
from skimage import filters
from scipy import ndimage as ndi
from skimage import morphology
from scipy.ndimage import gaussian_gradient_magnitude

#Load GIPL file
image_us, image_header = io.load('test_trus.gipl')

# #--------------Orthogonal Slice Example-----------------------
# image_slice= image_us[:,:,30]
# fig=plt.figure()
# plt.imshow(image_slice,cmap='gray')
# plt.show()

#--------------Non-Orthogonal Slice---------------------------
def reslice_volume(image, point, normal_vector):

    """
    Extracts a 2-D non-orthogonal slice from a 3-D volumetric data

    INPUTS:
    
    image: 3-D data of the volume - (datatype: numpy array)
    point: A point on the image volume - (datatype: tuple)
    normal_vector: A normal vector to the slicing plane of the image - (datatype: tuple)

    OUTPUTS:

    slice_image: A 2-D non-orthogonal slice from the 3-D image volume.
    
    """
    #Calculate the slicing plane using normal and vector
    d = - point[0] * normal_vector[0] - point[1] * normal_vector[1] - point[2] * normal_vector[2]
    xx, yy = np.meshgrid(range(image.shape[0]),range(image.shape[1]),indexing = "ij")
    zz = (-normal_vector[0] * xx - normal_vector[1] * yy - d) * 1.0 / normal_vector[2]
    zz = np.round(zz).astype(int)
    zz[zz < 0] = 0
    zz[zz >= image.shape[2] ] = image.shape[2] - 1
    slice_image = image[xx, yy, zz]  

    return slice_image 

#--------------Function to display non-orthogonal-----------------
def slice_plot(slice_image):
    """
    Plots a 2-D non-orthogonal slice on a 3D plot space

    INPUTS:
    
    slice_image: A 2-D non-orthogonal slice from the 3-D image volume.
    
    """
    print(slice_image.shape)
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111,projection="3d")
    x, y = np.meshgrid(range(slice_image.shape[1]),range(slice_image.shape[0]))
    ax.plot_surface(x, y, slice_image, cmap='gray')
    plt.show()

# def plot_3d_matrix(slice_image, angle=45):
#     x, y = np.meshgrid(range(slice_image.shape[1]), range(slice_image.shape[0]))
#     z = np.sin(np.deg2rad(angle)) * np.sqrt(x**2 + y**2)
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot_surface(x, y, z)
#     plt.show()


#------------2D BILATERAL FILTER-------------------------
def gaussianfilterfn(a,sigma):
    """
    Gaussian function for 2D bilateral filter function. Remove high frequency component from the image.

    INPUTS:
    
    a: Value for plotting/calculating gaussian 
    sigma: standard deviation 

    OUTPUTS:

    k: Output from the Gaussian equation
    
    """

    #Gaussian equation
    k = 0
    k = (1.0/(2*np.pi*(sigma)))*np.exp(-(a**2)/(2*(sigma**2)))
    return k

#Euclidean Distance
def euclideandistance2d(x0,y0,x1,y1):
    """
    Euclidean distance function for 2D

    INPUTS:
    
    x0: Point value across x 
    y0: Point value across y
    x1: Point value across x
    y1: Point value across y

    OUTPUTS:

    edistance2d: Output of euclidean distance
    
    """

    #Euclidean equation
    edistance2d = 0
    edistance2d = np.sqrt(np.abs((x0-x1)**2+(y0-y1)**2))
    return edistance2d


def bilateral_filter2d(input_image, d, sigma_range2d, sigma_domain2d):
    """
    Bilateral filter for 2D re-sliced image. Contains two sigma values.

    INPUTS:
    
    input_image: resliced 2D image - (type: Numpy array)
    d: Neighbourhood diameter, ideally leass than 7 - (datatype: int)
    sigma_range2d: Standard deviation of the range filter - (datatype: float)
    sigma_domain2d: Standard deviation of the range filter - (datatype: float)

    OUTPUTS:

    new_filter: filtered image output - (type:Numpy array)
    
    """
    start = time.time()
    #Pre-allocate values for the filtered image array
    new_filter = np.zeros(input_image.shape)

    #Iteration across two dimensions of the image
    for r in range(input_image.shape[0]):
        for c in range(input_image.shape[1]):
            total_weight = 0
            filtered_image = 0
            #Iteration across neighbourhood diameter d
            for i in range(d):
                for j in range(d):
                    #Difference across each point in row and column 
                    xx =r - (d/2 - i)
                    yy =c - (d/2 - j)
                    #New dimensions for computing gaussian kernel
                    if xx >= input_image.shape[0]:
                        xx = xx - 455
                    if yy >= input_image.shape[1]:
                        yy = yy - 325
                    #Gaussian kernels for convolution
                    g_range2d = gaussianfilterfn(input_image[int(xx)][int(yy)] - input_image[r][c], sigma_range2d)
                    g_domain2d = gaussianfilterfn(euclideandistance2d(xx, yy, r, c), sigma_domain2d)
                    # Convolution
                    wp = g_range2d * g_domain2d
                    filtered_image = (filtered_image) + (input_image[int(xx)][int(yy)] * wp)
                    total_weight = total_weight + wp
            filtered_image = filtered_image // total_weight
            new_filter[r][c] = int(np.round(filtered_image))

    stop = time.time()
    #Prints time difference from start to end operation of the function
    print('Timed value of the function:',stop-start)           
    return new_filter


#-----------------COMPARISON EXPERIMENTS-------------------
#MSE
def MSE(A,B):
    """
    To calculate Mean Square Error(MSE)

    INPUTS:
    
    A: resliced 2D image - (type: Numpy array with float values)
    B: filtered resliced 2D image - (type: Numpy array with float values)

    OUTPUTS:

    error: The output of MSE - (datatype:float)
    
    """
	# MSE
	# sum of the squared difference between the two images;
	# the two images must have the same dimension
    error = np.sum((A-B)**2)
    error = error/float(A.shape[0]*A.shape[1])
	
	# return the MSE
    return error

def compare_images(A,B,title):
    """
    Plot the comparison of two images - Original and Bilateral images and compute MSE and SSIM

    INPUTS:
    
    A: resliced 2D image - (type: Numpy array with float values)
    B: filtered resliced 2D image - (type: Numpy array with float values)
    title: Title of the figure - (type:string)

    OUTPUT:

    Displays the comparison image

    
    """
	# compute the mean squared error and structural similarity
	# index for the images
	#MSE
    m = MSE(A,B)
    #SSIM   
    s = ssim(A,B)
	# setup the figure
    fig = plt.figure(title)
    plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
	# show first image
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(A, cmap = 'gray')
    plt.axis("off")
    # show the second image
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(B, cmap = 'gray')
    plt.axis("off")
    # show the images
    plt.show()


def master():

    #Load GIPL file
    image_us, image_header = io.load('/home/alanb/cwdata/test_trus.gipl')

    #--------------Orthogonal Slice Example-----------------------
    image_slice= image_us[:,:,30]
    fig=plt.figure()
    plt.imshow(image_slice,cmap='gray')
    plt.show()

    #-----------Non-Orthogonal Slice Example-----------------------
    points = (150,200,40)
    normal_vectors = (0,10,10)
    slice_image = reslice_volume(image=image_us,point=points,normal_vector=normal_vectors)
    print(slice_image)
    plt.figure()
    plt.imshow(slice_image,cmap='gray')
    # plt.savefig('/home/alanb/cwdata/SLICE_bilateralan.png',dpi=300,bbox_inches='tight')
    # np.save('/home/alanb/cwdata/SLICE_bilateralan.npy',np.array(slice_image))
    slice_plot(slice_image=slice_image)

    #-----------2D re-sliced Bilateral Filter-----------------------
    kernel_size = 7
    # print(kernel_size)
    bilateral_image = bilateral_filter2d(input_image=slice_image, d= kernel_size, sigma_range2d= 5 , sigma_domain2d=20)
    #Image slice converted to float
    new_slice = img_as_float(slice_image)
    print(bilateral_image) 
    plt.figure()
    plt.imshow(bilateral_image,cmap='gray')
    plt.show()


    #----------COMPARISON EXPERIMENTS--------------------------
    
    # MSE AND SSIM       
    compare_images(A=new_slice,B=bilateral_image, title = 'Quantitative Comparison')

    #Gaussian Gradient Magnitude of original re-sliced and filtered image
    oimage2d = gaussian_gradient_magnitude(slice_image,sigma=5)
    gimage2d = gaussian_gradient_magnitude(bilateral_image,sigma=5)

    # # Histograms of original re-sliced and filtered image
    # ax = plt.hist(new_slice.ravel(),bins=256)
    # plt.show()
    # bx = plt.hist(bilateral_image.ravel(), bins=256)
    # plt.show()
    #Mean
    print('Mean of slice image:',np.mean(slice_image))    
    print('Mean of resliced bilateral image:',np.mean(bilateral_image))  
    #Standard Deviation 
    print('SD of slice image:',np.std(slice_image))    
    print('SD of resliced bilateral image:',np.std(bilateral_image))

    #CANNY EDGE DETECTOR USING SKIMAGE
    slice_edge= feature.canny(slice_image,sigma=3)
    bilateral_edge= feature.canny(bilateral_image,sigma=3)

    fig=plt.figure()
    bx = fig.add_subplot(1, 2, 1)
    plt.imshow(slice_edge, cmap = 'gray')
    plt.axis("off")
    bx = fig.add_subplot(1, 2, 2)
    plt.imshow(bilateral_edge, cmap = 'gray')
    plt.axis("off")
    # show the images
    plt.show()

    #Plot for report
    fig=plt.figure(figsize=(6,6))
    bx = fig.add_subplot(1, 6, 1)
    bx.set_title('(a)')
    plt.imshow(slice_image, cmap = 'gray')
    plt.axis("off")
    bx = fig.add_subplot(1, 6, 2)
    bx.set_title('(b)')
    plt.imshow(bilateral_image, cmap = 'gray')
    plt.axis("off")
    bx = fig.add_subplot(1, 6, 3)
    bx.set_title('(c)')
    plt.imshow(slice_edge, cmap = 'gray')
    plt.axis("off")
    bx = fig.add_subplot(1, 6, 4)
    bx.set_title('(d)')
    plt.imshow(bilateral_edge, cmap = 'gray')
    plt.axis("off")
    bx = fig.add_subplot(1, 6, 5)
    bx.set_title('(e)')
    plt.imshow(oimage2d, cmap = 'gray')
    plt.axis("off")
    bx = fig.add_subplot(1, 6, 6)
    bx.set_title('(f)')
    plt.imshow(gimage2d, cmap = 'gray')
    plt.axis("off")

    # Histograms of original re-sliced and filtered image
    fig=plt.figure(figsize=(4,4))
    bx = fig.add_subplot(1, 2, 1)
    bx.set_title('(a)')
    plt.hist(slice_image.ravel(),bins=256)
    # plt.axis("off")
    bx = fig.add_subplot(1, 2, 2)
    bx.set_title('(b)')
    plt.hist(bilateral_image.ravel(),bins=256)
    # plt.axis("off")
    plt.show()

    # Gaussian Gradient Magnitudes
    fig=plt.figure(figsize=(4,4))
    bx = fig.add_subplot(1, 2, 1)
    bx.set_title('(a)')
    plt.imshow(oimage2d,cmap='gray')
    plt.axis("off")
    bx = fig.add_subplot(1, 2, 2)
    bx.set_title('(b)')
    plt.imshow(gimage2d,cmap='gray')
    plt.axis("off")
    plt.show()


