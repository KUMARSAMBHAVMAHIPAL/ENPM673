#!/usr/bin/env python

import numpy as np
import cv2
from time import sleep

import math as m
import sys, os
# from numpy.linalg import inv,norm

def sq_detector_hull(contour, hierarchy,num=1):
    ids = list(range(len(contour)))
    ids = sorted(ids, 
    key= lambda i: cv2.contourArea(contour[i])/(0.001+cv2.contourArea(cv2.convexHull(contour[i],True))))
    
    orcnt = ids[:num]
    sqcnt = [hierarchy[0,i][-1] for i in orcnt]
    return sqcnt


def sq_detector_area(contour, hierarchy, min_area=1000, max_area=30000, num = 3):
    cnts =list(i for i in contour if min_area< cv2.contourArea(i)< max_area)
    sqcnts = [cv2.approxPolyDP(i, 0.02*cv2.arcLength(i,True), True) for i in cnts]
    fin = list(i for i in sqcnts if len(i)==4)[:num]
    return fin
            

def reorient(img):
    if img[2][2]==1 and img[2][5]==0 and  img[5][5]==0 and img[5][2]==0 :
        return cv2.rotate(img,cv2.ROTATE_180)
    elif img[2][2]==0 and img[2][5]==1 and  img[5][5]==0 and img[5][2]==0 :
        return cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
    elif img[2][2]==0 and img[2][5]==0 and  img[5][5]==0 and img[5][2]==1 :
        return cv2.rotate(img,cv2.ROTATE_90_COUNTERCLOCKWISE)
    else: return img
    
def rotate_lena(img):
    lena = cv2.imread('./data/reference_images/Lena.png')
    if img[2][2]==1 and img[2][5]==0 and  img[5][5]==0 and img[5][2]==0 :
        return cv2.rotate(lena,cv2.ROTATE_180)
    elif img[2][2]==0 and img[2][5]==1 and  img[5][5]==0 and img[5][2]==0 :
        return cv2.rotate(lena,cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif img[2][2]==0 and img[2][5]==0 and  img[5][5]==0 and img[5][2]==1 :
        return cv2.rotate(lena,cv2.ROTATE_90_CLOCKWISE)
    else: return lena

def tag_id(image):

    AR_tag = cv2.resize(image, (8,8))//255
    img = reorient(AR_tag)
    return int(8*img[4][3]+4*img[4][4]+2*img[3][4]+img[3][3])
                       
def angle(a,b):
    angle = np.arctan2(a[0]-b[0],a[1]-b[1])*180/np.pi
    if angle<0:
        return 360 + angle
    else:
        return angle

def detect_corner_position(sqcnt):
    if len(sqcnt)==4:
        sqcnt = np.array(sqcnt).flatten().reshape((4,2))
    elif len(sqcnt)>4:
        sqcnt = np.array(sqcnt[:4]).flatten().reshape((4,2))
    else:
        sqcnt = np.zeros((4,2))
    centroid = np.mean(sqcnt, axis=0)
    sqcnt=sorted(sqcnt, key= lambda i: angle(i,centroid))
    sqcnt = np.array(sqcnt).flatten().reshape((4,2))
    return sqcnt


def homography(sqcnt,size = 199):
    
    orig_order = np.array(
            [
                [size,0],
                [0,0],
                [0,size],
                [size,size],
            ])

    
    x1 = sqcnt[0,0]
    y1 = sqcnt[0,1]
    xp1 = orig_order[0,0]
    yp1 = orig_order[0,1]

    x2 = sqcnt[1,0]
    y2 = sqcnt[1,1]
    xp2 = orig_order[1,0]
    yp2 = orig_order[1,1]

    x3 = sqcnt[2,0]
    y3 = sqcnt[2,1]
    xp3 = orig_order[2,0]
    yp3 = orig_order[2,1]

    x4 = sqcnt[3,0]
    y4 = sqcnt[3,1]
    xp4 = orig_order[3,0]
    yp4 = orig_order[3,1]

    A = np.array([
        [-x1,-y1,-1,0,0,0,x1*xp1,y1*xp1,xp1],
        [0,0,0,-x1,-y1,-1,x1*yp1,y1*yp1,yp1],
        [-x2,-y2,-1,0,0,0,x2*xp2,y2*xp2,xp2],
        [0,0,0,-x2,-y2,-1,x2*yp2,y2*yp2,yp2],
        [-x3,-y3,-1,0,0,0,x3*xp3,y3*xp3,xp3],
        [0,0,0,-x3,-y3,-1,x3*yp3,y3*yp3,yp3],
        [-x4,-y4,-1,0,0,0,x4*xp4,y4*xp4,xp4],
        [0,0,0,-x4,-y4,-1,x4*yp4,y4*yp4,yp4],
        ],dtype="float64")
    
    u,s,v =np.linalg.svd(A)
    H= np.linalg.inv(v[-1].reshape(3,3))
    H = H/H[-1,-1]
    return H

def unwarp(img, H, size=200):
    unwarped = np.zeros((size,size))
    
    for i in range(size):
        for j in range(size):
            x,y,z = np.matmul(H,[i,j,1])
            if 0< (x//z) <1920 and 0< (y//z)< 1080:
                unwarped[i][j]=img[int(y//z)][int(x//z)]
    return unwarped

def projection_matrix(h, K):
    h1 = h[:,0]
    h2 = h[:,1]
    h3 = h[:,2]
    lamda = 2 / (np.linalg.norm(np.matmul(np.linalg.inv(K),h1)) + np.linalg.norm(np.matmul(np.linalg.inv(K),h2)))
    B = lamda * np.matmul(np.linalg.inv(K),h)

    if np.linalg.det(B)<0:
        B=-B
    r1 = B[:, 0]
    r2 = B[:, 1]
    r3 = np.cross(r1, r2)
    t = B[:, 2]
    R = np.column_stack((r1, r2, r3, t))
    P_matrix = np.matmul(K,R)
    return P_matrix

def projection_points(P):
    pts = [
        [199,0,0,1],
        [0,0,0,1],
        [0,199,0,1],
        [199,199,0,1],
        [199,0,-199,1],
        [0,0,-199,1],
        [0,199,-199,1],
        [199,199,-199,1]
    ]
    
    pts = [np.matmul(P,i) for i in pts]    
    return pts

def draw(img, pts):
    pts = np.array([[[int(i[0]/i[-1]),int(i[1]/i[-1])]] for i in pts])
    img = cv2.drawContours(img,[pts[:4]],-1,(0,0,255),3)
    img = cv2.drawContours(img,[pts[4:]],-1,(255,0,0),3)
    for i in range(4):
        img = cv2.line(img, tuple(pts[i][0]),tuple(pts[i+4][0]), (0,255,0),3)
    return img
    
K =np.array([[1406.08415449821,0,0],[2.20679787308599, 1417.99930662800,0],[1014.13643417416, 566.347754321696,1]]).T


def check_id(filepath, outfile=None):
    cap = cv2.VideoCapture(filepath)
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    if outfile != None:
        out = cv2.VideoWriter(outfile, cv2.VideoWriter_fourcc(*'MP4V') , 15, (1920,1080))
    count=0
    while(cap.isOpened()):
        try:
            ret, frame = cap.read()
#             print(frame.shape)
            num=1
            count+=1
            print(count, end='\r')
            frame2 = cv2.GaussianBlur(frame,(9,9),1.7)
            gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            ret, threshold = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY)
            contour, heir = cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

            cnt = [cv2.approxPolyDP(i,0.01*cv2.arcLength(i,True),True) for i in contour]

            sqcnt = sq_detector_area(contour,heir, num=num)
            frame = cv2.drawContours(frame, sqcnt,-1,(0,0,255),2)
            size = 200
            sqcnt = sqcnt[0].flatten().reshape((4,2))
#             print(sqcnt)
            H = homography(sqcnt,size=size)

            unwarped = unwarp(threshold,H)
            n = tag_id(unwarped)
            cv2.putText(frame,bin(n)[2:]+': '+str(n), (300,300), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255))

            if outfile!= None:
                out.write(frame)
            else:
                cv2.imshow('Frame', frame)
                cv2.imshow('frame', unwarped)
    #         print(n)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        except IndexError as e:
                pass
        except Exception as e:
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            break
        

    cap.release()
    if outfile!= None:
        out.release()
    cv2.destroyAllWindows()
    
    
def lena(filepath= './data/Video_dataset/multipleTags.mp4', outfile=None):
    
    if filepath=='./data/Video_dataset/multipleTags.mp4':
        num = 3
    else: num=1
        
    cap = cv2.VideoCapture(filepath)
    if outfile != None:
        out = cv2.VideoWriter(outfile, cv2.VideoWriter_fourcc(*'MP4V') , 15, (1920,1080))
    count=0
    if (cap.isOpened()== False): 
        print("Error opening video stream or file") 
    print(num)
    
    while(cap.isOpened()):
        try:
            count+=1
            print(count, end='\r')
            ret, frame = cap.read()
            frame2 = cv2.GaussianBlur(frame,(9,9),1.7)
            gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            ret, threshold = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY)
            contour, heir = cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
            size = 200
            sqcnt = sq_detector_area(contour,heir, num=num,min_area=1000, max_area=7000)

            corners = [[detect_corner_position(i)] for i in sqcnt]
            H = [homography(i[0],size=size-1) for i in corners]

            img1 = unwarp(threshold,H[0], size=size)
            artag = cv2.resize(img1, (8,8))//255
            lena1 = rotate_lena(artag)
            lena1 = cv2.resize(lena1,(size,size))

            img2 = unwarp(threshold,H[1],size=size)
            artag = cv2.resize(img2, (8,8))//255
            lena2 = rotate_lena(artag)
            lena2 = cv2.resize(lena2,(size,size))

            img3 = unwarp(threshold,H[2],size=size)
            artag = cv2.resize(img2, (8,8))//255
            lena3 = rotate_lena(artag)
            lena3 = cv2.resize(lena3,(size,size))

            for i in range(size):
                for j in range(size):
                    x,y,z = np.matmul(H[0],[i,j,1])
                    frame[int(y//z),int(x//z),:]=lena1[i][j]
                    x,y,z = np.matmul(H[1],[i,j,1])
                    frame[int(y//z),int(x//z),:]=lena2[i][j]
                    x,y,z = np.matmul(H[2],[i,j,1])
                    frame[int(y//z),int(x//z),:]=lena3[i][j]

            if outfile!= None:
                out.write(frame)
            else:
                cv2.imshow('Frame', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        except IndexError as e:
            pass
        except Exception as e:
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            break

    cap.release()
    if outfile!= None:
        out.release()
    cv2.destroyAllWindows()
    
def cube_placement(filepath, outfile=None):
    if filepath=='./data/Video_dataset/multipleTags.mp4':
        num = 3
    else: num=1
    
    cap = cv2.VideoCapture(filepath)
    
    if outfile != None:
        out = cv2.VideoWriter(outfile, cv2.VideoWriter_fourcc(*'MP4V') , 15, (1920,1080))
    count=0
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")    
    while(cap.isOpened()):
        try:
            ret, frame = cap.read()
            count+=1
            print(count, end='\r')
            frame2 = cv2.GaussianBlur(frame,(9,9),1.7)
            gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            ret, threshold = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY)
            contour, heir = cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
            size = 199
            if num==3:
                min_area = 1000
                max_area = 9000
            sqcnt = sq_detector_area(contour,heir, num=num, min_area=min_area, max_area=max_area)
            corners = [detect_corner_position(i) for i in sqcnt]
#             frame = cv2.drawContours(frame, corners,-1,(0,0,255),1)
            H = [homography(i, size=size) for i in corners]
            
#             print(H[0])
            for i in range(num):
                try:
                    proj= projection_matrix(H[i], K)
                    pts = projection_points(proj)
                    frame= draw(frame,pts)
                except Exception as e:
                    pass
#             print(1+'a')
            if outfile!= None:
                out.write(frame)
            else:
                cv2.imshow('Frame', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        except IndexError as e:
            pass
        except Exception as e:
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            break

    cap.release()
    if outfile!=None:
        out.release()
    cv2.destroyAllWindows()


if __name__=='__main':
	cube_placement('./data/Video_dataset/multipleTags.mp4', outfile = 'cube.mp4')
	lena(outfile = 'lena.mp4')
	check_id('./data/Video_dataset/Tag0.mp4', outfile= 'Tag0_id.mp4')
	check_id('./data/Video_dataset/Tag1.mp4', outfile= 'Tag1_id.mp4')
	check_id('./data/Video_dataset/Tag2.mp4', outfile= 'Tag2_id.mp4')