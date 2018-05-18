# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
usage: change your own data dir and it works
"""

import numpy as np
import os
from collections import namedtuple
from PIL import Image
import cv2
import matplotlib.pylab as plt
import codecs


Img_dataset_dir = 'E:/zfeng/tianchi/sheshuang/train_1000/image_1000'
Label_dataset_dir = 'E:/zfeng/tianchi/sheshuang/train_1000/txt_1000'
crop_dataset_dir_horiz = 'E:/zfeng/tianchi/sheshuang/Crop_dataset_1000/train_1000/horizontal/'
crop_dataset_dir_vert = 'E:/zfeng/tianchi/sheshuang/Crop_dataset_1000/train_1000/vertical/'

Image_list = os.listdir(Img_dataset_dir)
Label_list = os.listdir(Label_dataset_dir)

def get_txt_label(label_path):
    coordinates = []
    labels = []
    with open (label_path,encoding='utf-8')  as f:
        for line in f.readlines():

            coordinate = line.split(',')[0:8]
            label = line.split(',')[-1].strip()
            coordinates.append(coordinate)
            labels.append(label)

    return coordinates,labels

def transform(x1,y1,x2,y2,x3,y3,x4,y4):
    height1 = np.sqrt((x1-x4)**2 + (y4-y1)**2)
    height2 = np.sqrt((x3-x2)**2 + (y3-y2)**2)
    h = max(height1,height2)

    width1 = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    width2 = np.sqrt((x3-x4)**2 + (y3-y4)**2)
    w = max(width1,width2)

    Pts = np.float32(np.array([[0,0],[w,0],[w,h],[0,h]]))#顺时针
    return Pts,w,h
for i in range(len(Image_list)):
    img = Image.open(os.path.join(Img_dataset_dir,Image_list[i])).convert('RGB')
#    plt.imshow(img)
    coordinates,labels = get_txt_label(os.path.join(Label_dataset_dir,Label_list[i]))
    coordinates = np.array(coordinates)
#    print(len(coordinates))
#    print(len(labels))
#    print(Image_list[i])
    for j in range(coordinates.shape[0]):
        coord= namedtuple('coord',['x1','y1','x2','y2','x3','y3','x4','y4'])
        coordinate = coord(coordinates[j][0],coordinates[j][1],coordinates[j][2],coordinates[j][3],coordinates[j][4],coordinates[j][5],coordinates[j][6],coordinates[j][7])
        label =labels[j]
        if label == str('###'):
            pass
#            new.save(os.path.join(crop_no_use_dir,name))
#         if label==str('"###"'):
#             pass
        else:
            X = list(map(float,[coordinate.x1,coordinate.x2,coordinate.x3,coordinate.x4]))
            Y = list(map(float,[coordinate.y1,coordinate.y2,coordinate.y3,coordinate.y4]))

            Xmin = min(X)
            Xmax = max(X)
            Ymin = min(Y)
            Ymax = max(Y)

            Pts1 = np.float32(np.array([[X[0],Y[0]],[X[1],Y[1]],[X[2],Y[2]],[X[3],Y[3]]]))
            Pts2,W,H = transform(X[0],Y[0],X[1],Y[1],X[2],Y[2],X[3],Y[3])
            M = cv2.getPerspectiveTransform(Pts1,Pts2)
            img1 = np.array(img)
            Dst = cv2.warpPerspective(img1,M,(int(W),int(H)))
            img_new = Image.fromarray(Dst)
            # plt.imshow(Dst)
            # plt.show()

            name = str(str(i)+'_'+str(j)+'.jpg')
            if img_new.size[0]>=1.2*img_new.size[1]:#(0为宽，1位高)#横的
                """对new做resize"""
                p=img_new.size[1]/31
                if p==0:
                    continue
                new_height=int(img_new.size[1]/p)
                new_width=int(img_new.size[0]/p)
                new_0=img_new.resize((new_width,new_height))
                try:
                    new_0.save(os.path.join(crop_dataset_dir_horiz, name))
                    f = codecs.open(os.path.join(crop_dataset_dir_horiz,'label_horiz.txt'),'a',encoding='utf-8')
                    f.write(str(crop_dataset_dir_horiz+name+' '+label+'\n'))
                    f1 = codecs.open(os.path.join(crop_dataset_dir_horiz,'label_ciku.txt'),'a',encoding='utf-8')
                    f1.write(label+'\n')
                except:
                    continue
            else:#竖的图片
                p = img_new.size[0]/31
                if p==0:
                    continue
                new_height=int(img_new.size[1]/p)
                new_width=int(img_new.size[0]/p)
                new_1=img_new.resize((new_width,new_height))
#               print(label)
                try:
                    new_1.save(os.path.join(crop_dataset_dir_vert, name))

                    f = codecs.open(os.path.join(crop_dataset_dir_vert,'label.txt'),'a',encoding='utf-8')
                    f.write(str(crop_dataset_dir_vert+name+' '+label+'\n'))
                    f1 = codecs.open(os.path.join(crop_dataset_dir_vert,'label_ciku.txt'),'a',encoding='utf-8')
                    f1.write(label+'\n')

                except:
                    continue
            f.close()
            f1.close()
#    plt.close()