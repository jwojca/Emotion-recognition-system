import cv2
import random
import numpy as np
import os
import winsound

from playsound import playsound

def horizontal_flip(img, flag):
    if flag:
        return cv2.flip(img, 1)
    else:
        return img

def brightness(img, low, high):
    value = random.uniform(low, high)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype = np.float64)
    hsv[:,:,1] = hsv[:,:,1]*value
    hsv[:,:,1][hsv[:,:,1]>255]  = 255
    hsv[:,:,2] = hsv[:,:,2]*value 
    hsv[:,:,2][hsv[:,:,2]>255]  = 255
    hsv = np.array(hsv, dtype = np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img

def fill(img, h, w):
    img = cv2.resize(img, (w, h), cv2.INTER_CUBIC)
    return img
        
def horizontal_shift(img, ratio=0.0):
    if ratio > 1 or ratio < 0:
        print('Value should be less than 1 and greater than 0')
        return img
    ratio = random.uniform(-ratio, ratio)
    h, w = img.shape[:2]
    to_shift = w*ratio
    if ratio > 0:
        img = img[:, :int(w-to_shift), :]
    if ratio < 0:
        img = img[:, int(-1*to_shift):, :]
    img = fill(img, h, w)
    return img

def channel_shift(img, value):
    value = int(random.uniform(-value, value))
    img = img + value
    img[:,:,:][img[:,:,:]>255]  = 255
    img[:,:,:][img[:,:,:]<0]  = 0
    img = img.astype(np.uint8)
    return img

def squeeze(img, lowPct, highPCt):
    h, w = img.shape[:2]
    wSqueezed = int(w * (1 - (random.uniform(lowPct, highPCt)/100)))
    img = cv2.resize(img, (wSqueezed, h), interpolation = cv2.INTER_AREA)
    return img

def expand(img, lowPct, highPCt):
    h, w = img.shape[:2]
    wAddition = int(w * (random.uniform(lowPct, highPCt)/100))
    img = cv2.resize(img, (w + wAddition, h), interpolation = cv2.INTER_AREA)
    return img

def rotation(img, maxAngle):
    # angle = int(random.uniform(-angle, angle))
    angle = random.randint(int(maxAngle*0.6), maxAngle)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))
    return img

inputDir = r'C:\Users\hwojc\Desktop\Diplomka\AffectNet\Train\surprise'
outputDir = r'C:\Users\hwojc\Desktop\Diplomka\AffectNet\Train\surprise\augmented'

maxAngle = 6
maxBrightness = 1.5
minBrightness = 0.5

origOn = True
flipOn = True
brightnessOn = True
channelShiftOn = False
grayOn = True
hShiftOn = False
squeezeOn = True
expandOn = True


for images in os.listdir(inputDir):
    if(images.endswith(".jpg") or images.endswith(".png")):
        path = os.path.join(inputDir,images)
        img = cv2.imread(path)
        filenameBase = images.removesuffix('.png')

        if(origOn):
            cv2.imwrite(os.path.join(outputDir, images), img)

        # Write flipped image
        if(flipOn):
            imgFlipped = horizontal_flip(img, True)
            filename = filenameBase + '_Flipped' + '.png'
            cv2.imwrite(os.path.join(outputDir, filename), imgFlipped)

        if(brightnessOn):
            # Write original image with different brightness
            imgChBrigthOrig = brightness(img, minBrightness, maxBrightness)
            imgChBrigthOrig = rotation(imgChBrigthOrig, maxAngle)
            filename = filenameBase + '_ChBright' + '.png'
            cv2.imwrite(os.path.join(outputDir, filename), imgChBrigthOrig)

            # Write flipped image with different brightness
            imgFlipped = horizontal_flip(img, True)
            imgChBrigthFlipped = brightness(imgFlipped, minBrightness, maxBrightness)
            imgChBrigthFlipped = rotation(imgChBrigthFlipped, maxAngle)
            filename = filenameBase + '_ChBrightFlipped' + '.png'
            cv2.imwrite(os.path.join(outputDir, filename), imgChBrigthFlipped)

        if(hShiftOn):
            # Write horizontaly shifted image
            imgHShifted = horizontal_shift(img, 0.2)
            filename = filenameBase + '_HShifted' + '.png'
            cv2.imwrite(os.path.join(outputDir, filename), imgHShifted)
       
        
        if(channelShiftOn):
            # Write chanell shifted image
            imgChannShift = channel_shift(img, 20)
            imgChannShift = rotation(imgChannShift, maxAngle)
            filename = filenameBase + '_ChannShift' + '.png'
            cv2.imwrite(os.path.join(outputDir, filename), imgChannShift) 

        if(grayOn):
            imgGrayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            imgGrayscale = rotation(imgGrayscale, maxAngle)
            filename = filenameBase + '_Gray' + '.png'
            cv2.imwrite(os.path.join(outputDir, filename), imgGrayscale) 

        if(squeezeOn):
            imgSqeezed = squeeze(img, 10, 30)
            imgSqeezed = rotation(imgSqeezed, maxAngle)
            filename = filenameBase + '_Sqeezed' + '.png'
            cv2.imwrite(os.path.join(outputDir, filename), imgSqeezed) 
        
        if(expandOn):
            imgExpanded = expand(img, 10, 30)
            imgExpanded = rotation(imgExpanded, maxAngle)
            filename = filenameBase + '_Expanded' + '.png'
            cv2.imwrite(os.path.join(outputDir, filename), imgExpanded) 
        
    


print('Successfully saved')

winsound.PlaySound("SystemQuestion", winsound.SND_ALIAS)
