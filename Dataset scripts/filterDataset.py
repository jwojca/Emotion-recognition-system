import cv2
import os
import csv

#change path of images
pathImg = r'C:\Users\hwojc\Desktop\Revize datasetu Affect Net\disgust rev'

#change path of CSV file
pathCSV = r'C:\Users\hwojc\Desktop\Revize datasetu Affect Net\disgust rev\toDelete.csv'

arr = []
with open(pathCSV, newline='') as csvfile:
    for row in csv.reader(csvfile):
        arr.append(', '.join(row))



for i in range(0,len(arr)):
    for images in os.listdir(pathImg):
        if(images.endswith(".jpg") or images.endswith(".png")):
            pathImgRemove = os.path.join(pathImg,images)
            if(images == arr[i]):
                os.remove(pathImgRemove)


print("DONE")
cv2.destroyAllWindows()


 
