from os import listdir
from os.path import isfile, join
import numpy 
from numpy import array
import cv2
import scipy.io as sio
import h5py
import csv
with open('./train_sorted.csv','r') as f1:
  reader=csv.reader(f1,delimiter=',')
  index=0
  index2=1
  count=1001
  header=next(reader)
  filename="reduced"+str(index)+"k_"+str(index2)+"k"+".csv"
  quit=0
  start=0
  while(quit==0):
    with open(filename,'w') as f2:
      writer = csv.writer(f2)
      writer.writerows([header])
      if(start==1):
        writer.writerows([temprow])
      temprow=next(reader)
      while(int(temprow[2])!=count):
        writer.writerows([temprow])
        try:
          temprow=next(reader)
        except:
          quit=1
          break
    count=count+1000
    index=index+1 
    index2=index2+1 
    start=1
    filename="reduced"+str(index)+"k_"+str(index2)+"k"+".csv"
