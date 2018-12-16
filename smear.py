import os
import argparse
from skimage import io
from skimage import measure
from skimage.morphology import closing,square
import numpy as np
if __name__ == '__main__':
    #only image regions greater than MAX_AREA pixels will be smeared
    MAX_AREA = 400
    parser = argparse.ArgumentParser()
    parser.add_argument('-s','--saliency',required = True,help="the direcroty of saliency map")
    parser.add_argument('-d','--dataset',required = True,help="the direcroty of image dataset")
    parser.add_argument('-t','--threshold',type=int,required = True,help="the threshold of image binarization.the value of threshold is between 0 to 255")
    parser.add_argument('-o','--output',required = True,help="dictionary to store images after binarization")
    args = parser.parse_args()  
    
    if not os.path.exists(args.output):
        os.mkdir(args.output)  

    #read image from saliency path
    for f in os.listdir(args.saliency):
        filename = os.path.join(args.saliency,f)
        
        image = io.imread(filename)
        #print (image)
        #the values of image are between 0 to 255 
        #image binarization
        binary_img = image > args.threshold
        binary_img = binary_img * 255

        #close small holes with binary closing
        img_closing = closing(binary_img,square(3))
        
        #prepare output image
        mask = np.zeros(image.shape,dtype=np.uint8)
        org_name = f[:-3] + 'jpg'
        org_img = io.imread(os.path.join(args.dataset,org_name)) 


        #find the connected regions, and draw their bounding box
        label = measure.label(img_closing, connectivity =2)
        regions = measure.regionprops(label)
        
        for region in regions:
            #only image regions greater than MAX_AREA pixels will be smeared
            if region.area > MAX_AREA:
                minr,minc,maxr,maxc = region.bbox
                org_img[minr:maxr,minc:maxc,:] = 255
                mask[minr:maxr,minc:maxc] = 255


        #write the image in output directory
        io.imsave(os.path.join(args.output,'sm_'+f),org_img)

        io.imsave(os.path.join(args.output,'mask'+f),mask)  

        io.imsave(os.path.join(args.output,'binary'+f),img_closing)








      
        
