import sys
import os
import cv2
import time 
import math
import numpy 
import os
import shutil
import subprocess
from PIL import Image
from datetime import datetime 
from tqdm import tqdm
import numpy as np 
from matplotlib import pyplot as plt 


def checkpath():
    if len(sys.argv) != 2:
        print ("Only one parameter can be entered")
    else:
        inputpath = sys.argv[-1]
        if not os.path.exists(inputpath):
            print ("Error input path")
        else:
            if inputpath.split('/')[-1].split('.')[-1] == "mp4":
                parent_path = os.path.dirname(inputpath) + '/pictest'
                if os.path.exists(parent_path):
                    shutil.rmtree(parent_path)
                os.makedirs(parent_path)
                duration = os.popen('mediainfo '+inputpath+' |grep Duration |awk -F \':|s\' \'NR==1{print$2}\'').read().replace('\n','').replace('\r','')
                return inputpath,parent_path,int(duration)
            else:
                print ("Error input file,Please input the mp4 file")
    
def vidtopic1(videopath,outputpath):
    vc=cv2.VideoCapture(videopath)
    c=1  
    allframe = int(vc.get(7))
    if vc.isOpened():  
        rval,frame=vc.read()  
    else:  
        rval=False  
        print ("Can not open the mp4 file")
    print ('picture processing')
    pbar = tqdm(total=allframe,)
    while rval:
        rval,frame=vc.read()
        try:
            frame.shape
        except BaseException:
            continue
        else:
            #frame = frame[0:int(vc.get(4)),int(vc.get(3))/2:int(vc.get(3))]
            #frame = cv2.resize(frame,(1080,1920),interpolation=cv2.INTER_CUBIC)
            cv2.imwrite((outputpath+'/picture-'+str("%03d" % c)+'.jpg'),frame)
            pbar.update(1)
            c=c+1 
    pbar.close()
    vc.release()

def Hamming_distance(hash1,hash2): 
    num = 0 
    for index in range(len(hash1)): 
        if hash1[index] != hash2[index]: 
            num += 1 
    return num 

def classify_aHash(image1,image2): 
    image1 = cv2.resize(image1,(8,8)) 
    image2 = cv2.resize(image2,(8,8)) 
    gray1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY) 
    gray2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY) 
    hash1 = getHash(gray1) 
    hash2 = getHash(gray2) 
    return Hamming_distance(hash1,hash2) 

def getHash(image): 
    avreage = np.mean(image) 
    hash = [] 
    for i in range(image.shape[0]): 
        for j in range(image.shape[1]): 
            if image[i,j] > avreage: 
                hash.append(1) 
            else: 
                hash.append(0) 
    return hash 
    
def count(picpath,duration):
    pathDir =  os.listdir(picpath)
    fps = 0
    picsum = len(pathDir)
    print ('count processing')
    pbar = tqdm(total=picsum)
    for piccount in range(1,len(pathDir)):
        image1 = os.path.join(picpath,pathDir[piccount-1])
        image2 = os.path.join(picpath,pathDir[piccount])
        vc1=cv2.imread(image1)
        vc2=cv2.imread(image2)
        distance = classify_aHash(vc1,vc2)
        if distance != 0:
            fps = fps + 1
        pbar.update(1)
    perfps = int(fps)/duration
    return fps,perfps,duration
        

if __name__ == '__main__':
    picturepath = checkpath()
    if picturepath != None:
        vidtopic1(picturepath[0], picturepath[1])
        finalfps = count(picturepath[1],picturepath[2])
        print ('allframe: ' +  str(finalfps[0]))
        print ('fps: ' +  str(finalfps[1]))
        print ('duration: ' + str(finalfps[2]))


