import os
import argparse
from skimage import io
from skimage import measure
from skimage.morphology import closing,square
import numpy as np
import random
from MMCQ import MMCQ
if __name__ == '__main__':
    #only image regions greater than MAX_AREA pixels will be smeared
    MAX_AREA = 400
    parser = argparse.ArgumentParser()
    parser.add_argument('-s','--saliency',required = True,help="the direcroty of saliency map")
    parser.add_argument('-d','--dataset',required = True,help="the direcroty of image dataset")
    parser.add_argument('-t','--threshold',type=int,required = True,help="the threshold of image binarization.the value of threshold is between 0 to 255")
    parser.add_argument('-f','--fill',required = True,choices = ['white','random','color'],help="the color to fill the privacy region")    
    parser.add_argument('-o','--output',required = True,help="dictionary to store images after binarization")
    args = parser.parse_args()  
    
    if not os.path.exists(args.output):
        os.mkdir(args.output)  
    mask_path = os.path.join(args.output,'mask')
    pic_path = os.path.join(args.output,'picture')
    if not os.path.exists(mask_path):
        os.mkdir(mask_path)
    
    if not os.path.exists(pic_path):
        os.mkdir(pic_path)
    
    
    #read image from saliency path
    for f in os.listdir(args.saliency):
        filename = os.path.join(args.saliency,f)
        #read saliency map
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

        #prepare the color for filling
        if (args.fill == 'color'):
            #extract the theme color
            maxColor = 2
            mmcq = MMCQ(org_img, maxColor)
            theme= mmcq.quantize()
            #fill_content=[R,G,B]
            fill_content = theme[0]
        elif (args.fill == 'white'):
            fill_content = [255,255,255]
        

        #find the connected regions, and draw their bounding box
        label = measure.label(img_closing, connectivity =2)
        regions = measure.regionprops(label)
        
        for region in regions:
            #only image regions greater than MAX_AREA pixels will be smeared
            if region.area > MAX_AREA:
                minr,minc,maxr,maxc = region.bbox
                if (args.fill == 'random'):
                    org_img[minr:maxr,minc:maxc] =np.random.randint(0,256,(maxr-minr,maxc-minc,3),np.uint8)
                else:                
                    org_img[minr:maxr,minc:maxc] = fill_content
                
                mask[minr:maxr,minc:maxc] = 255


        #write the image in output directory
        io.imsave(os.path.join(pic_path,f),org_img)

        io.imsave(os.path.join(mask_path,f),mask)  

        #io.imsave(os.path.join(args.output,'binary'+f),img_closing)








      
        
