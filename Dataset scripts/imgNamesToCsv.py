import os
import csv

#change path of images
pathImg = r'C:\Users\hwojc\OneDrive - Vysoké učení technické v Brně\Magisterské studium\Diplomka\04 Datasety\AffectNet\Merged\disgust'

#change path of CSV file
pathCSV = r'C:\Users\hwojc\Desktop\Revize datasetu Affect Net\disgust rev\toDelete.csv'

f = open(pathCSV, 'w')


for images in os.listdir(pathImg):
        f.write(images)
        f.write('\n')
        
print("DONE")        
        