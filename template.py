import cv2 as cv
import numpy as np
import os

#################################################################

# Write a function Convolve (I, H). I is an image of varying size, H is a kernel of varying size.
# The output of the function should be the convolution result that is displayed.

def convolve(I, H):

    # Reads the image
    image = I

    # Initiallizes the col and row size of kernal H
    kernCols = 0
    kernRows = 0
    
    #print("hiiiiii", H.shape)
    

    if len(H.shape) == 1:
        # Handles the 1D kernal that has 1 row (1, c)
        kernCols = len(H)
        kernRows = 1
    else:
        kernCols = len(H[0])
        kernRows = len(H)


    width, height, channel = image.shape
    convolvedImage = np.empty((height, width, 3))



    # Loops through each pixel in the image
    for h in range(height):
        for w in range(width):

            # Initializes the new pixel value
            pixel = np.zeros(3)

            # Loops through each pixel around the current pixel, corresponding to the size of the kernel
            for row in range(h-kernRows//2, h+kernRows//2+1):
                for col in range(w-kernCols//2, w+kernCols//2+1):

                    # Handles boundaries by excluding the border pixels
                    if (not(col < 0 or col>height-1 or row<0 or row>width-1)):

                        if len(H.shape) == 1:
                            # Handles the 1D kernal that has 1 row (1, c)
                            pixel += (image[row][col] * H[h+kernRows//2-row])
                        else:
                            # Applies each kernal value to each of the surrounding pixels, 
                            # to determine the center pixel's new value
                            pixel += (image[row][col] * H[h+kernRows//2-row][w+kernCols//2-col])

            # Updates the pixel in the convolvedImage   
            convolvedImage[h, w] = pixel

    # Loops through each pixel in the image
    for h in range(height):
        for w in range(width):
            
            # Loops through the colors of the pixel
            # and updates it depending on the value
            for color in range(len(convolvedImage[h][w])):
                if convolvedImage[h][w][color] < 0:
                    convolvedImage[h][w][color] = 0
                elif convolvedImage[h][w][color] > 1:
                    convolvedImage[h][w][color] /= 255
    
    return convolvedImage




#################################################################

# Write a function Reduce(I) that takes image I as input and outputs a copy of the image resampled
# by half the width and height of the input. Remember to Gaussian filter the image before reducing it; 
# use separable 1D Gaussian kernels.

def reduce(I):

    # Reads the image
    #image = cv.imread(I)

    # Uses seperable 1D Gaussian kernals and uses the Gaussian filter on the image
    x = np.array([1, 2, 1])/4
    y = np.array([[1],[2],[1]])/4
    gausA = convolve(I, x)/2
    gausB = convolve(I, y)/2
    final = gausA + gausB
    #blurImg = cv.GaussianBlur(image, (5, 5), 0)


    # Uses resize() function to reduce the image by half its width and height
    return cv.resize(final, None, fx = 0.5, fy = 0.5, interpolation=cv.INTER_AREA)




#################################################################

# Write a function Expand(I) that takes image I as input and outputs a copy of the image expanded, 
# twice the width and height of the input.

def expand(I):

    # Uses resize() function to expand the image by twice its width and height
    return cv.resize(I, None, fx = 2, fy = 2, interpolation=cv.INTER_LINEAR)



#################################################################

# Use the Reduce() function to write the GaussianPyramid(I,n) function, where n is the no. of levels.

def gaussianPyramid(I, n):

   
    # Creates an array to store each image in the pyramid, starting with image I
    gausPyramid = [I]

    # Loop through n times
    for i in range(n):

        # Uses reduce() function and adds the resulting image to the array
        I = reduce(I)
        gausPyramid.append(I)
    
    # Returns the pyramid (array of images)
    return gausPyramid

#################################################################

# Use the above functions to write LaplacianPyramids(I,n) that produces n level Laplacian pyramid of I.

def laplacianPyramid(I, n):

    # Use conolve on the image
    I = convolve(I, np.array([[0, 0, 0],[0, 1, 0],[0, 0, 0]]))

    # Use gaussianPyramid() function and store images of the gaussian pyramid in an array
    gausPyr = gaussianPyramid(I, n)

    # Creates an array to store each image in the Laplacian pyramid, 
    # starting with smallest img from the gaus pry
    laplPyramid = [gausPyr[len(gausPyr)-1]]

    # Loop through n times in reverse
    for i in reversed(range(n)):

        # Calls the subtracts the current image of the gaus pyramid form the
        # expanded image of the period gaus image 
        img = gausPyr[i] - expand(gausPyr[i+1])
        laplPyramid.append(img)

    # Returns array of images of the laplacian pyramid
    return laplPyramid




#################################################################

# Write the Reconstruct(LI,n) function which collapses the Laplacian pyramid LI of n levels 
# to generate the original image. Report the error in reconstruction using image difference.

def reconstruct(LI, n):

    # Initializes the new image
    newImg = LI[1] + expand(LI[0])

    # Loops through the remainging images in the Laplacian pyramid
    for i in range(2,n+1):
        newImg = LI[i] + expand(newImg)

    return newImg



def mosaic(laplPyrA, laplPyrB, AImg, BImg, n):

    # Add the left and right sides of each pyramid
    LS = []
    for la,lb in zip(laplPyrA, laplPyrB):
        rows,cols,dpt = la.shape
        ls = np.hstack((la[:,0:cols//2], lb[:,cols//2:]))
        LS.append(ls)
    
    # Reconstruct the image
    ls_ = LS[0]
    for i in range(1,n):
        ls_ = cv.pyrUp(ls_)
        ls_ = cv.add(ls_, LS[i])
    
    # Create final mosaic image with both left and right side
    mosaicImg = np.hstack((AImg[:,:cols//2],BImg[:,cols//2:]))

    return mosaicImg
        

#################################################################


def main():

    # Reads the image, which will be passed into each function
    image = cv.imread("images/lena.png")
    
    # Problem 1: Convolve
    convolveImage = convolve(image, np.array([[1, 2, 1],[0, 0, 0],[-1, -2, -1]]))

    # Problem 2: Reduce
    reduceImg = reduce(image)

    # Problem 3: Expand
    expandImg = expand(image)   

    # Problem 4: Gaussian Pyramid        
    gausPyrImg = gaussianPyramid(image, 3)

    # Problem 5: Laplacian Pyramid
    laplPyrImg = laplacianPyramid(image, 3)

    # Problem 6: Reconstruct
    reconstructImg = reconstruct(laplPyrImg, 3)
    # Error in reconstruction:
    errorImg = (image/255) - reconstructImg
    print(errorImg)


    # Problem 7: Mosaicking

    # Mosaic Test A
    testA1Img = cv.imread("images/Test_A1.png")
    testA2Img = cv.imread("images/Test_A2.png")
    newA0Img = cv.resize(testA1Img, (512, 512))
    newB0Img = cv.resize(testA2Img, (512, 512))
    laplPyrA1 = laplacianPyramid(newA0Img, 3)
    laplPyrA2 = laplacianPyramid(newB0Img, 3)
    mosaicImgA = mosaic(laplPyrA1, laplPyrA2, newA0Img, newB0Img, 3)


     # Mosaic Test B
    testB1Img = cv.imread("images/Test_B1.png")
    testB2Img = cv.imread("images/Test_B2.png")
    newA1Img = cv.resize(testB1Img, (512, 512))
    newB1Img = cv.resize(testB2Img, (512, 512))
    laplPyrB1 = laplacianPyramid(newA1Img, 3)
    laplPyrB2 = laplacianPyramid(newB1Img, 3)
    mosaicImgB = mosaic(laplPyrB1, laplPyrB2, newA1Img, newB1Img, 3)

    # Mosaic Test C
    testC1Img = cv.imread("images/Test_C1.png")
    testC2Img = cv.imread("images/Test_C2.png")
    newA2Img = cv.resize(testC1Img, (512, 512))
    newB2Img = cv.resize(testC2Img, (512, 512))
    laplPyrC1 = laplacianPyramid(newA2Img, 3)
    laplPyrC2 = laplacianPyramid(newB2Img, 3)
    mosaicImgC = mosaic(laplPyrC1, laplPyrC2, newA2Img, newB2Img, 3)

    # Mosaic Test D
    testD1Img = cv.imread("images/Test_D1.png")
    testD2Img = cv.imread("images/Test_D2.png")
    newA3Img = cv.resize(testD1Img, (512, 512))
    newB3Img = cv.resize(testD2Img, (512, 512))
    laplPyrD1 = laplacianPyramid(newA3Img, 3)
    laplPyrD2 = laplacianPyramid(newB3Img, 3)
    mosaicImgD = mosaic(laplPyrD1, laplPyrD2, newA3Img, newB3Img, 3)
 
   

    
     # Shows Original Image
    cv.imshow('Original Image', image)
    cv.imshow('Convolved Image', convolveImage)
    cv.imshow('Reduced Image', reduceImg)
    cv.imshow('Expanded Image', expandImg)

    
    # Shows Gaussian Pyramid
    cv.imshow('Gaussian Pyramid Image 0', gausPyrImg[0])
    cv.imshow('Gaussian Pyramid Image 1', gausPyrImg[1])
    cv.imshow('Gaussian Pyramid Image 2', gausPyrImg[2])
    cv.imshow('Gaussian Pyramid Image 3', gausPyrImg[3])

    # Shows Laplacian Pyramid
    cv.imshow('Laplacian Pyramid Image 0', laplPyrImg[0])
    cv.imshow('Laplacian Pyramid Image 1', laplPyrImg[1])
    cv.imshow('Laplacian Pyramid Image 2', laplPyrImg[2])
    cv.imshow('Laplacian Pyramid Image 3', laplPyrImg[3])

    # Shows reconstructed image
    cv.imshow('Reconstructed Image', reconstructImg)

    #cv.imshow('Error of Reconstructed Image', errorImg)

    # Shows the 4 mosaic images
    cv.imshow('Mosaic Image for Test A', mosaicImgA)
    cv.imshow('Mosaic Image for Test B', mosaicImgB)
    cv.imshow('Mosaic Image for Test C', mosaicImgC)
    cv.imshow('Mosaic Image for Test D', mosaicImgD)

    # Saves the 4 mosaic images
    cv.imwrite("images/TestA.png", mosaicImgA)
    cv.imwrite("images/TestB.png", mosaicImgB)
    cv.imwrite("images/TestC.png", mosaicImgC)
    cv.imwrite("images/TestD.png", mosaicImgD)


    cv.waitKey(0)
    cv.destroyAllWindows()


    

if __name__ == '__main__':
    main()