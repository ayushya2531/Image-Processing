from PIL import ImageTk
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import Tkinter as tk
from Tkinter import *
from tkFileDialog import askopenfilename

global filename




def upload_image():
    global filename
    print (upload_image.__name__)
    filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
    print(filename)
    novi = tk.Toplevel()
    canvas = tk.Canvas(novi, width = 600, height = 600)
    canvas.pack(expand = YES, fill = BOTH)
    im = PIL.Image.open(filename)
    gif1 = ImageTk.PhotoImage(im)
    canvas.create_image(10, 10, image = gif1, anchor = NW)
    canvas.gif1 = gif1


def red_eye(): ### Credit : PyTech Solutions, Learn-OpenCV.com
    

    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    imgOut = img.copy()

    eyesCascade = cv2.CascadeClassifier("/home/ayushya/Downloads/RedEyeRemover/haarcascade_eye.xml")
    eyes = eyesCascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(50, 50))

    for ind,(x, y, w, h) in enumerate(eyes):

        eye = img[y:y+h, x:x+w,:]
        b = eye[:, :, 0]
        g = eye[:, :, 1]
        r = eye[:, :, 2]

        bg = cv2.add(b, g)

        # Simple red eye detector.
        mask = (r > 100) &  (r > bg)

        mask = mask.astype(np.uint8)*255

        maskFloodfill = mask.copy()
        maskTemp = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(maskFloodfill, maskTemp, (0, 0), 255)
        mask2 = cv2.bitwise_not(maskFloodfill)
        mask = mask2 | mask

        mask = cv2.dilate(mask, None, anchor=(-1, -1), iterations=3, borderType=1, borderValue=1)

        contours, _ = cv2.findContours(mask.copy() ,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)   
        maxArea = 0
        maxCont = None
        for cont in contours:
            area = cv2.contourArea(cont)
            print(area)
            if area > maxArea:
                maxArea = area
                maxCont = cont
        mask = mask * 0 # Reset the mask image to complete black image
        cv2.drawContours(mask , [maxCont],0 ,(255),-1 )
        mask[np.nonzero(mask)] = 1
        m = np.logical_not(mask).astype(np.uint8)

        r_1half = np.multiply(m,r)
        r_2half = np.multiply(mask,bg/2)
        eye[:,:,2] = r_1half + r_2half
        imgOut[y:y+h,x:x+w,:] = eye
    
    imgOut1 = imgOut[:,:,[2,1,0]]
    #print(imgOut1.shape)
    I_o = PIL.Image.fromarray(imgOut1,'RGB')
    #print(I_o)
    novi1 = tk.Toplevel()
    canvas1 = tk.Canvas(novi1, width = 600, height = 600)
    canvas1.pack(expand = YES, fill = BOTH)
    gif1 = ImageTk.PhotoImage(I_o)
    canvas1.create_image(10, 10, image = gif1, anchor = NW)
    canvas1.gif1 = gif1

    #cv2.imshow('Red Eyes Removed', imgOut)
    #cv2.waitKey(0)

    
def de_noise():
    print (de_noise.__name__)
    
    img = cv2.imread(filename, 0)
    img1 = img.copy()
    w,h = img.shape

    m_im = np.mean(img)
    if m_im<127:
        gamma = 0.7
    else:
        gamma = 2.0
    img2 = img1/255.0
    img2 = cv2.pow(img2, gamma)
    img2 = (img2*255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    equ = clahe.apply(img2)
    equ_c = equ.copy()
    cv2.fastNlMeansDenoising(equ, equ_c, 3.0, 7, 21);
    gaussian_3 = cv2.GaussianBlur(equ_c, (9,9), 10.0)
    unsharp_image = cv2.addWeighted(equ_c, 1.5, gaussian_3, -0.5, 0, equ_c)


    
    I_de = PIL.Image.fromarray(unsharp_image,'L')
    novi2 = tk.Toplevel()
    canvas2 = tk.Canvas(novi2, width = 550, height = 600)
    canvas2.pack(expand = YES, fill = BOTH)
    gif2 = ImageTk.PhotoImage(I_de)
    canvas2.create_image(10, 10, image = gif2, anchor = NW)
    canvas2.gif2 = gif2


root = Tk()
root.geometry("1000x1200")
root.resizable(0,0)
root.title("Image Processing")
root.configure(background='black')
background_image=PhotoImage(file = '/home/ayushya/Downloads/bg.png')
background_label = Label(root, image=background_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)



button_1 = Button(root, text = "Upload Image", fg = "yellow", bg = "black", command = upload_image)
button_2 = Button(root, text = "Red-Eye Correction", fg = "yellow", bg = "black", command = red_eye)
button_3 = Button(root, text = "Autoenhancement", fg = "yellow", bg = "black", command = de_noise)
button_1.place(x=400, y=200, relwidth=0.2, relheight = 0.05)
button_2.place(x=200, y=600, relwidth=0.2, relheight = 0.05)
button_3.place(x=600, y=600, relwidth=0.2, relheight = 0.05)


root.mainloop()