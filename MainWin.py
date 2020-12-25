# coding:utf-8
import os
import numpy as np
import cv2
import time

from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow
from uiM import *
import sys

from tensorflow.python.platform import gfile
import tensorflow as tf
from PIL import Image

from pathlib import Path


sess = tf.Session()
pb_file_path = "./pb/my-model-36002_yu2.pb"


with gfile.FastGFile(pb_file_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

sess.run(tf.local_variables_initializer())
sess.run(tf.global_variables_initializer())

sess2 = tf.Session()
#pb_file_path2="./pb/best-22.pb"
pb_file_path2="./pb/besttf1.6-47_v1202.pb"#"./pb/besttf-7_v1120.pb"#"./pb/win5_subimg_average@2019.11.19-18.28.41_bestAcc.ckpt-7.pb"
with gfile.FastGFile(pb_file_path2, 'rb') as f2:
    graph_def2 = tf.GraphDef()
    graph_def2.ParseFromString(f2.read())
    sess2.graph.as_default()
    tf.import_graph_def(graph_def2, name='')

sess2.run(tf.local_variables_initializer())
sess2.run(tf.global_variables_initializer())

N=40
def saveimagesnumpy(imlist,sdir):
    num = 0
    for im in imlist:
        im = Image.fromarray(im.astype("uint8"))
        num = num + 1
        im.save(sdir + '/' + str(num) + '.png')

def saveimages(imlist,sdir):
    num=0
    for im in imlist:
        num=num+1
        im.save(sdir + '/' + str(num) + '.png')
def cvimwrite1(cropImg,savedir):
    # print("cropimg:::",cropImg)
    h, w= cropImg.shape
    #print(cropImg.shape)
    global num

    num = num + 1
    cv2.imwrite(savedir+"/{}.jpg".format(num), cropImg)



def pinjie_hang(imlist, W, H):
    #print(W)
    #if (W<55):
    #    print("-----------------------------------------------------------:",W)
        #break
    target = Image.new('RGB', (W, H))
    x1 = 0
    x2 = 0
    for im in imlist:
        w, h = im.size
        x1 = x2
        x2 = x2 + w
        target.paste(im, (x1, 0, x2, h))
    return target
def pinjie_lie(imlist,W,H):
    target = Image.new('RGB', (W, H))
    y1=0
    y2=0
    for im in imlist:

        w,h=im.size
        y1=y2
        y2=y2+h
        target.paste(im,(0,y1,w,y2))
    return target

def input_pipeline(pinlist,pinlist1):
    n_tuple=N
    batch_size=1
    img_width=64
    img_height=64
    inChannel=2

    batch_images = np.zeros((batch_size, n_tuple, img_width, img_height, inChannel), dtype=np.float32)

    batch_labels = np.zeros((batch_size, 2), dtype=np.float32)
    for j in range(n_tuple):
            img=pinlist[j]
            img = img.convert('L')
            w, h = img.size
            if (w != 64 or h != 64):
                img = img.resize((64, 64))
            curr_image = np.array(img, dtype=np.float32)
            if len(img.getbands()) != 1:
                print("ERROR image contains wrong number channel...")
            img1 = pinlist1[j]
            img1 = img1.convert('L')
            
            w1, h1 = img1.size
            if (w1 != 64 or h1 != 64):
                img1 = img1.resize((64, 64))
            curr_image1 = np.array(img1, dtype=np.float32)
           
            if len(img1.getbands()) != 1:  # self.inChannel:
                print("ERROR image contains wrong number channel...", file_name1)
            batch_images[0, j, :, :, 0] = (curr_image)  # /255.0) #- self.meanPixel
            batch_images[0, j, :, :, 1] = (curr_image1)  # /255.0) #- self.meanPixel
    return batch_images#, label_batch



def getHProjection(image):
    hProjection = np.zeros(image.shape, np.uint8)
    # 图像高与宽
    (h, w) = image.shape
    # 长度与图像高度一致的数组
    h_ = [0] * h
    # 循环统计每一行白色像素的个数
    for y in range(h):
        for x in range(w):
            if image[y, x] == 255:
                h_[y] += 1
    # 绘制水平投影图像
    for y in range(h):
        for x in range(h_[y]):
            hProjection[y, x] = 255
    #cv2.imshow('hProjection2', hProjection)
    #cv2.waitKey(0)

    return h_


def getVProjection(image):

    x1_=0
    x2_=0
    vProjection = np.zeros(image.shape, np.uint8);
    # 图像高与宽
    (h, w) = image.shape
    # 长度与图像宽度一致的数组
    w_ = [0] * w
    # 循环统计每一列白色像素的个数
    for x in range(w):
        for y in range(h):
            if image[y, x] == 255:
                w_[x] += 1
    if(w<10):
        return

    for x in range(w-10):
        if(w_[x+10]>0 and w_[x+10]<h):
            x1_=x+10
            break
    for x in range(w+10):
        #print(w,"wx",x)
        if(w_[w-x-11]>0 and w_[w-x-11]<h):
            x2_=w-x-11
            break

    # 绘制垂直平投影图像
    for x in range(w):
        for y in range(h - w_[x], h):
            vProjection[y, x] = 255
    '''cv2.namedWindow("vProjection", 0)
    cv2.resizeWindow("vProjection", 640, 480);
    cv2.imshow('vProjection', vProjection)
    cv2.waitKey(0)'''

    return w_,x1_,x2_
def MaxW(im_list):
    print("come")
    r_list = []
    if(len(im_list)>N):


        im_list.sort(key=lambda ele: ele[0])

        for im in im_list[:40]:
            r_list.append(im[1])
    else:
        for im in im_list:
            r_list.append(im[1])
    return r_list


def cv_imread(file_path):
    cv_img=cv2.imdecode(np.fromfile(file_path,dtype=np.uint8), 1)
    return cv_img


class MyImageWindow(QMainWindow, Ui_Form):

    def __init__(self, parent = None):

        super(MyImageWindow, self).__init__(parent)
        self.setupUi(self)

        self.filepath1 = ""
        self.filepath2 = ""

        self.pinlist1=[]
        self.pinlist2=[]
        self.a_pinlist=[]
        self.b_pinlist = []

        self.img = tf.placeholder(dtype=tf.float32, shape=[None, None], name='img_')
        self.lei = tf.placeholder(dtype=tf.int64, shape=[1], name='lei_')
        cv_tensor = tf.image.convert_image_dtype(self.img, tf.float32)
        cv_tensor = tf.expand_dims(cv_tensor, -1)

        new_size = tf.constant([64, 64], dtype=tf.int32)
        self.cv_tensor = tf.image.resize_images(cv_tensor, new_size)#, name="cv_tensor_")


    
    def cls_img_yuanshi(self,imgarr, k):####上一个版本的

        imlist=[]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, k)
        eroded = cv2.erode(imgarr, kernel, iterations=2)
        retval, gray_temp = cv2.threshold(eroded, 180, 255, cv2.THRESH_BINARY_INV)
        contours, hierarchy = cv2.findContours(gray_temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        num = 0

        for i in range(0, len(contours)):
            x, y, w, h = cv2.boundingRect(contours[i])

            if h > 15 and h < 330:  # and h<100 and w>10 and w<100:# and w>10:

                cropImg = imgarr[y:y + h, x:x + w]

                while (w > 55):
                    # w=w-30
                    cropImg1 = cropImg[:, :30]
                    num = num + 1

                    h,w=cropImg1.shape
                    if (w < 20 or h < 20 or h > 60 or w > 60):
                        continue
                    imlist.append(cropImg1)

                    cropImg = cropImg[:, 30:]
                    h, w = cropImg.shape
                num = num + 1

                h,w = cropImg.shape
                if (w < 20 or h < 20 or h > 60 or w > 60):
                    continue

                imlist.append(cropImg)
        return  imlist


    def cls_img(self,imgarr, k):

        imlist=[]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, k)
        eroded = cv2.erode(imgarr, kernel, iterations=2)
        retval, gray_temp = cv2.threshold(eroded, 180, 255, cv2.THRESH_BINARY_INV)
        contours, hierarchy = cv2.findContours(gray_temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        num = 0

        for i in range(0, len(contours)):
            x, y, w, h = cv2.boundingRect(contours[i])

            if h > 15 and h < 430 :  # and h<100 and w>10 and w<100:# and w>10:

                cropImg = imgarr[y:y + h, x:x + w]

                while (w > 55):
                    # w=w-30
                    cropImg1 = cropImg[:, :30]
                    #num = num + 1


                    while h>55:
                        cropImg2=cropImg1[:30,:]
                        num=num+1

                        imlist.append(cropImg2)
                        cropImg1=cropImg1[30:,:]
                        h, w = cropImg1.shape

                    h, w = cropImg1.shape
                    # if (w < 20 or h < 20 or h > 100 or w > 100):
                    #if (w < 20 or h < 20 or h > 100 or w > 100):
                    #    continue
                    if (h>15):
                        num=num+1
                        imlist.append(cropImg1)

                    cropImg = cropImg[:, 30:]
                    h, w = cropImg.shape
                #num = num + 1

                h,w = cropImg.shape
                #if (w < 20 or h < 20 or h > 100 or w > 100):
                #    continue
                if w>15:
                    while h > 55:
                        cropImg2 = cropImg[:30, :]
                        num = num + 1

                        imlist.append(cropImg2)
                        cropImg = cropImg[30:, :]
                        h, w = cropImg.shape
                    h, w = cropImg.shape
                    if h > 15:
                        num=num+1
                        imlist.append(cropImg)

        return  imlist
    def getHVProjection(self,orImage, savedir):

        Position = []
        # 水平投影
        retval, img = cv2.threshold(orImage, 180, 255, cv2.THRESH_BINARY_INV)
        (h, w) = img.shape
        H = getHProjection(img)

        start = 0
        H_Start = []
        H_End = []
        # 根据水平投影获取垂直分割位置
        for i in range(len(H)):
            if H[i] > 65 and start == 0:
                H_Start.append(i)
                start = 1
            if H[i] <= 65 and start == 1:
                H_End.append(i)
                start = 0
        # 分割行，分割之后再进行列分割并保存分割位置
        for i in range(len(H_Start) - 1):
            # 获取行图像
            # print("H_Start[i]::", H_Start[i], "H_End[i]::", H_End[i])
            cropImg = img[H_Start[i]:H_End[i], 0:w]
            # cv2.imshow('cropImg',cropImg)
            # cv2.waitKey(0)
            # 对行图像进行垂直投影
            W, _, _ = getVProjection(cropImg)
            Wstart = 0
            Wend = 0
            W_Start = 0
            W_End = 0
            for j in range(len(W)):
                if W[j] > 2 and Wstart == 0:
                    W_Start = j
                    Wstart = 1
                    Wend = 0
                if W[j] <= 2 and Wstart == 1:
                    W_End = j
                    Wstart = 0
                    Wend = 1
                if Wend == 1:
                    cropImg = orImage[H_Start[i] - 5:H_End[i], W_Start - 3:W_End + 3]
                    # print("H::", H_End[i] - H_Start[i], "W::", W_End - W_Start)
                    if (W_End - W_Start <= 11):
                        continue
                    if (H_End[i] - H_Start[i] <= 19):
                        continue

                    cvimwrite1(cropImg,savedir)

                    Wend = 0



    def pinjie(self, list_im):

        i = 0
        W = 0
        H = 0
        wlist = []
        im_hang_list = []
        llielist = []
        im_lie_list = []

        Wlie = 0
        Hlie = 0
        llielist = []
        im_hang_list = []

        while i < len(list_im):
            # im=list_im[i]
            imarr = 255 - list_im[i]
            im = Image.fromarray(imarr.astype("uint8"))

            w, h = im.size

            if w < 60:
                W = W + w
                if H < h:
                    H = h
                wlist.append(im)
                if (W > 65):
                    im_hang = pinjie_hang(wlist, W, H)
                    im_hang_list.append(im_hang)
                    if (H < 60):
                        Hlie = Hlie + H
                        if (Wlie < W):
                            Wlie = W
                        llielist.append(im_hang)
                        if (Hlie > 65):
                            im_lie = pinjie_lie(llielist, Wlie, Hlie)
                            im_lie_list.append((Wlie,im_lie))
                            Wlie = 0
                            Hlie = 0
                            llielist = []
                            im_hang_list = []

                    else:
                        im_lie_list.append((W,im_hang))

                    W = 0
                    H = 0
                    wlist = []
            else:
                im_hang_list.append(im)
                if (h < 60):
                    Hlie = Hlie + h
                    if (Wlie < w):
                        Wlie = w
                    llielist.append(im)
                    if (Hlie > 65):
                        im_lie = pinjie_lie(llielist, Wlie, Hlie)


                        im_lie_list.append((Wlie,im_lie))
                        Wlie = 0
                        Hlie = 0
                        llielist = []
                        im_hang_list = []
                else:
                    im_lie_list.append((w,im))
            #if (len(im_lie_list) > N):
            #    break
            i = i + 1
        #if(len(im_lie_list)>40):
        #   MaxW_im_lie_list=MaxW(im_lie_list)
        #else:
        MaxW_im_lie_list=im_lie_list#MaxW(im_lie_list)


        return MaxW_im_lie_list#im_lie_list

    def bnt1_click(self):
        self.filepath1, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择字迹A的图像", './dataset', '*.png *.jpg *.bmp')
        if self.filepath1 is '':
            return
        self.textEdit.setText(self.filepath1)
        jpg = QtGui.QPixmap(self.filepath1)#.scaled(self.label.width(), self.label.height())
        self.label.setPixmap(jpg)

        print("begin1..")
        import time
        start = time.clock()

        if self.filepath1 == "" :#or self.filepath2 == "":
            print("none")
            return

        cv_list = []

        print(self.filepath1)

        imsrc = cv_imread(self.filepath1)

        try:
            image = cv2.cvtColor(imsrc, cv2.COLOR_BGR2GRAY)

        except:
            image = imsrc
            #image2 = imsrc2
        k = (2, 2)


        imagelist=self.cls_img(image, k)

        saveimagesnumpy(imagelist, './b')



        #savefenge = "./fengge1"
        global num
        num = 0
        #self.getHVProjection(image, savefenge)

        for imgx in  imagelist:
            img = sess.run(self.cv_tensor, feed_dict={self.img: imgx})
            cv_list.append(img)

        input_x = sess.graph.get_tensor_by_name('image_batch:0')
        keep_prob = sess.graph.get_tensor_by_name('keep_prob:0')
        out_y = sess.graph.get_tensor_by_name('ArgMax:0')
        predlist1=[]
        for test_images in cv_list:
            ret = sess.run(out_y, feed_dict={input_x: [test_images], keep_prob: 1.0})
            c = ret.tolist()
            predlist1 += c

        #print(predlist1)

        a_imglist = []
        #b_imglist = []
        savepath1 = './cls1'
        #savepath2 = './cls2'
        num1 = 0


        for i in range(0, len(predlist1)):
            if (predlist1[i]) == 1:
                a_imglist.append(imagelist[i])
                num1 = num1 + 1
                cv2.imwrite(savepath1 + "/{}.png".format(num1), imagelist[i])

        self.a_pinlist += self.pinjie(a_imglist)



        self.pinlist1=MaxW(self.a_pinlist) #按照w 排序取出前最大N=40个
        #print(self.pinlist1)

        sdir1 = "./pin1"
        saveimages(self.pinlist1, sdir1)
        print("======================", len(self.pinlist1))
        #print("=======================", len(pinlist2))
        end = time.clock()

        jN = N / 2  # 最少需要的笔迹数 可以调整

        if (len(self.pinlist1) < jN):  # 少于N/2 认为笔迹不足
            #self.label_3.setText('提取的字迹B不足！')
            self.label_3.setText('提取的字迹A不足 共：{}  此次提取耗时:{}'.format(str( len(self.pinlist1)),str(end - start)))

        if(len(self.pinlist1)>jN) and (len(self.pinlist1)<N):#大于N/2笔迹可以预测
            self.label_3.setText('提取的字迹A可以预测 共：{}  此次提取耗时:{}'.format(str( len(self.pinlist1)),str(end - start)))

        if (len(self.pinlist1) >=N):  # 大于N笔迹充分
            self.label_3.setText('提取的字迹A充分 共：{}  此次提取耗时:{}'.format(str( len(self.pinlist1)),str(end - start)))
            print("none!!")


    def bnt2_click(self):
        self.filepath2, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择字迹B的图像", './dataset', '*.png *.jpg *.bmp')
        if self.filepath2 is '':
            return
        self.textEdit_2.setText(self.filepath2)
        jpg = QtGui.QPixmap(self.filepath2)#.scaled(self.label_2.width(), self.label_2.height())
        self.label_2.setPixmap(jpg)

        import time
        start = time.clock()

        if  self.filepath2 == "":
            print("none")
            return

        cv_list = []

        print(self.filepath2)

        imsrc2 = cv_imread(self.filepath2)

        try:
            image2 = cv2.cvtColor(imsrc2, cv2.COLOR_BGR2GRAY)
        except:
            image2 = imsrc2
        k = (2, 2)

        imagelist2=self.cls_img(image2, k)
        savefenge="./fengge"
        global num
        num=0
        self.getHVProjection(image2,savefenge)

        for imgx in  imagelist2:
            img = sess.run(self.cv_tensor, feed_dict={self.img: imgx})
            cv_list.append(img)

        input_x = sess.graph.get_tensor_by_name('image_batch:0')
        keep_prob = sess.graph.get_tensor_by_name('keep_prob:0')
        out_y = sess.graph.get_tensor_by_name('ArgMax:0')

        predlist2=[]
        for test_images in cv_list:
            ret = sess.run(out_y, feed_dict={input_x: [test_images], keep_prob: 1.0})
            c = ret.tolist()
            predlist2 += c

        b_imglist = []

        savepath2 = './cls2'

        num2 = 0

        for i in range(0, len(predlist2)):
            if (predlist2[i]) == 1:
                    b_imglist.append(imagelist2[i])
                    num2 = num2 + 1
                    cv2.imwrite(savepath2 + "/{}.png".format(num2), imagelist2[i])

        #self.pinlist2 += self.pinjie(b_imglist)
        self.b_pinlist += self.pinjie(b_imglist)
        self.pinlist2 = MaxW(self.b_pinlist) #按照w 排序取出前最大N=40个

        sdir2 = "./pin2"

        saveimages(self.pinlist2, sdir2)

        print("=======================", len(self.pinlist2))

        jN = N / 2  # 最少需要的笔迹数 可以调整
        end=time.clock()
        if (len(self.pinlist2) < jN):  # 少于N/2 认为笔迹不足
            self.label_3.setText('提取的字迹B不足 共：{} 提取耗时:{}'.format(str( len(self.pinlist2)),str(end - start)))

        if(len(self.pinlist2)>jN ) and (len(self.pinlist1)<N) :#大于N/2笔迹可以预测
            self.label_3.setText('提取的字迹B可以预测 共：{} 提取耗时:{}'.format(str( len(self.pinlist2)),str(end - start)))

        if (len(self.pinlist2) >=N):  # 大于N笔迹充分
            self.label_3.setText('提取的字迹B充分 共：{}  提取耗时:{}'.format(str( len(self.pinlist2)),str(end - start)))
 

    def bnt4_click(self):



        print("======================",len(self.pinlist1))
        print("=======================",len(self.pinlist2))
        import time
        start = time.clock()
        jN=N/2   #最少需要的笔迹数 可以调整
        if (len(self.pinlist1) < jN or len(self.pinlist2) <jN):  # 少于N/2 认为笔迹不足
            self.label_3.setText('提取的字迹不足！')
            print("none!!")

            return


        plist1=self.pinlist1
        plist2=self.pinlist2
        if (len(plist1) >=jN and len(plist1) <N):
            print("not enoughA! im_lie_list+im_lie_list")
            while len(plist1)<N:
                plist1=plist1+plist1

        if (len(plist2) >=jN and len(plist2) <N):
            print("not enoughA! im_lie_list+im_lie_list")
            while len(plist2)<N:
                 plist2=plist2+plist2

        test_z = input_pipeline(plist1, plist2)
        in_x = sess2.graph.get_tensor_by_name('Placeholder:0')
        out_y = sess2.graph.get_tensor_by_name('ArgMax_1:0')
        out_d = sess2.graph.get_tensor_by_name('Sub:0')
        ret, d = sess2.run([out_y, out_d], feed_dict={in_x:test_z})##???problem
        pred=ret[0]

        distance=d[0][0]

        end = time.clock()
        print(pred,distance)
        print(end-start)


        self.label_3.setText('预测:{},distance:{},鉴别预测耗时:{}'.format(str(pred), str(distance),str(end-start)))

        self.pinlist1=[]###清空
        self.pinlist2=[]
        self.a_pinlist=[]
        self.b_pinlist=[]
        self.textEdit.setText("")
        jpg = QtGui.QPixmap()  # .scaled(self.label.width(), self.label.height())
        self.label.setPixmap(jpg)
        self.textEdit_2.setText("")
        self.label_2.setPixmap(jpg)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWin = MyImageWindow()
    myWin.show()
    sys.exit(app.exec_())