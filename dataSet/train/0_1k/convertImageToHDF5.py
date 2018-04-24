from os import listdir
from os.path import isfile, join
import numpy 
from numpy import array
import cv2
import scipy.io as sio
import h5py
import csv
import operator
mypath='images0_1k/'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
images = numpy.empty([32,32,3,len(onlyfiles)],dtype=int)
labels = numpy.empty([1,len(onlyfiles)],dtype=int)
with open('reduced0_1k.csv','r') as f:
  reader=csv.reader(f)
  n=0
  numberOfImageNotAvailable=0
  next(reader)
  for row in reader: 
    labels[0][n]=int(row[2])
    filename=row[0]+".jpg" 
    try:
      imtemp =cv2.imread(join(mypath,filename))
      im=cv2.resize(imtemp,(32,32))
      images[:,:,:,n]=im
      n=n+1
    except:
      print(filename,"notavailable")
      numberOfImageNotAvailable=numberOfImageNotAvailable+1
hf=h5py.File('data0_1k.h5','w')
hf.create_dataset('X',data=images)
hf.create_dataset('y',data=labels)
hf.close
print("successfully completed")
