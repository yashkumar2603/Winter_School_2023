import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os


def edge_mask(img, line_size, blur_value):
        gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        gray_blur = cv.medianBlur(gray, blur_value)

        edges = cv.adaptiveThreshold(gray_blur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, line_size, blur_value)
        return edges

def color_quantization(img, k):
    #transform the image.
        data = np.float32(img).reshape((-1, 3))

    #Determine criteria
        criteria  = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 20, 0.001)

    #implementing k means
        ret, label, center = cv.kmeans(data, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)

        result = center[label.flatten()]
        result = result.reshape(img.shape)

        return result

def cartoon(blurred):
        c = cv.bitwise_and(blurred, blurred, mask = edges)

        return c

vid = cv.VideoCapture(0)
while(True):
    ret, frame = vid.read()
    scale_percent = 175# percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
  
    # resize image
    frame_resized = cv.resize(frame, dim, interpolation = cv.INTER_AREA)
    #cv.imshow('frame', frame_resized)
    #if cv.waitKey(1) & 0xFF == ord('q'):
        #break
    
    img = cv.cvtColor(frame_resized, cv.COLOR_BGR2RGB)

    line_size, blur_value = 9,5
    edges = edge_mask(img, line_size, blur_value)

    

    img = color_quantization(img, k = 7)
    
    blurred = cv.bilateralFilter(img, d = 7, sigmaColor = 200, sigmaSpace = 200)

    cv.imshow('frame', cartoon(blurred))
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv.destroyAllWindows()