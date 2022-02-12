# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 11:09:08 2021

@author: 70950
"""

from __future__ import print_function
#from libtiff import TIFFfile, TIFFimage  #pylibtiff
#matplotlib.use('GTKAgg') #backends, need to run it first to force the plt.show() plot window appears before the code is done
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
#import scipy.io
import time
#import scipy.sparse as sparse
import math
import winsound
import ipdb
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

pp=0
global_M=np.array([])

font= cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale= 2
fontColor= (255,255,0)
lineType= 2

path = 'D:/enmergence and soil/corn/corn1/'
filename=glob.glob(os.path.join(path+"rotated_imgs/", '*.JPG'))

def match_images(img1, img2,ratio=0.6,feature_filter=19,feature_movement=1,feature_threshold=300,use_filter=0,remove_bad=0,removed_edge=1):
    """Given two images, returns the matches"""
    #detector = cv2.xfeatures2d.SIFT_create(4000, 3, 3,1,1)
    detector = cv2.xfeatures2d.SURF_create(feature_threshold, feature_filter, feature_filter,feature_movement,feature_movement)
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    kp1, desc1 = detector.detectAndCompute(img1, None)
    #cv2.imwrite("0005.jpg", img3)
    
    kp2, desc2 = detector.detectAndCompute(img2, None)
    #kp1[0].pt #the coodinate of the keypoint

    raw_matches = matcher.knnMatch(desc1, trainDescriptors = desc2, k = 2) # k will be the dimension of each match 
    #dmatch.queryIdx: This attribute gives us the index of the descriptor in the list of query descriptors (in our case, it’s the list of descriptors in the img1).
    #dmatch.trainIdx: This attribute gives us the index of the descriptor in the list of train descriptors (in our case, it’s the list of descriptors in the img2).
    #dmatch.distance: This attribute gives us the distance between the descriptors. A lower distance indicates a better match.
    
    
    if use_filter==1:
        kp_pairs= filter_matches(kp1, kp2, raw_matches,ratio)
    else:
        kp_pairs=raw_matches
    
    # remove false matches
    if remove_bad==1:
        M,good=RemoveBadMatching(kp_pairs,kp1,kp2,img1,img2,removed_edge)
    else:
        M=np.zeros((1,2))
        good=kp_pairs

                 
    return kp1,kp2,good,M

def drawMyMatches(img1,kp1,img2,kp2,good,flag=0):
    global pp
    img7 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
    for f in range(len(list(good))):
         pt=np.array(kp1[good[f][0].queryIdx].pt)
         print("kp1,"+str(f)+":"+str(pt))
         bottomLeftCornerOfText = (int(pt[0]),int(pt[1]))
         cv2.putText(img7,str(f), bottomLeftCornerOfText, font, fontScale,fontColor,lineType)
         pt=np.array(kp2[good[f][0].trainIdx].pt)
         bottomLeftCornerOfText = (int(pt[0])+img1.shape[1],int(pt[1]))
         print("kp2,"+str(f)+":"+str(bottomLeftCornerOfText))
         cv2.putText(img7,str(f), bottomLeftCornerOfText, font, fontScale,fontColor,lineType)
         
    #plt.imshow(img7),plt.show()     
    cv2.imwrite(path+"matching_result/goodResult"+str(pp)+".jpg", img7)
    
    if flag==0: #good result
        pp=pp+1

    return

def filter_matches(kp1, kp2, matches, ratio = 1):

    good = []
    
    while len(list(good))<3:
        #ratio=ratio+0.05      
        for m,n in matches:
           if m.distance <= ratio*n.distance:  #The main advantage using knnMatch is that you can perform a ratio test. So if the distances from one descriptor in descriptor1 to the two best descriptors in descriptor2 are similar it suggests that there are repetitive patterns in your images (e.g. the tips of a picket fence in front of grass). Thus, such matches aren't reliable and should be removed.
              good.append([m])
    
    print('good:'+str(len(list(good)))+'  ratio='+str(ratio))
    
    return good

def getMatchMatric(kp1, kp2,kp_pairs):
    
    src_pts = np.float32([ kp1[m[0].queryIdx].pt for m in kp_pairs ])
    dst_pts = np.float32([ kp2[m[0].trainIdx].pt for m in kp_pairs ])
         
    x0=int(round(np.mean(src_pts[:,0]-dst_pts[:,0]))) #just get the x and y trandform firstly, but the best fit would change in the rotationFit
    y0=int(round(np.mean(src_pts[:,1]-dst_pts[:,1])))
 
 
    M=np.zeros((1,2))
    M[0,0]=x0 
    M[0,1]=y0
   
    return M

def RemoveBadMatching(kp_pairs,kp1,kp2,img1,img2, removed_edge=1):
    max_threshold=1.1
    min_threshold=0.9
    
    slope=[]
    distance=[]
    
    good=kp_pairs.copy()
    good_remove=[]
    src_pts = np.float32([ kp1[m[0].queryIdx].pt for m in kp_pairs ])
    dst_pts = np.float32([ kp2[m[0].trainIdx].pt for m in kp_pairs ])
    
    if removed_edge==1:
        for f in range(len(kp_pairs)):
            x1,y1=np.array(kp1[kp_pairs[f][0].queryIdx].pt)
            x2,y2=np.array(kp2[kp_pairs[f][0].trainIdx].pt)
            if x1<250 or x2<250 or x1>4700 or x2>4700 or y1<280 or y2<280 or y1>3500 or y2>3500:
                good_remove.append(f)
        
    
    good_remove.sort(reverse=True)
    #ipdb.set_trace()
    
    for f in good_remove:
        del good[f]
    
    #ipdb.set_trace()
    
    
    for f in range(len(good)):
        x1,y1=np.array(kp1[good[f][0].queryIdx].pt)
        x2,y2=np.array(kp2[good[f][0].trainIdx].pt)
        slope.append((y2-y1)/(x2-x1))
        distance.append((x2 - x1)**2 + (y2 - y1)**2)
        
    slope=np.array(slope)
    #good=kp_pairs.copy()
    
    for f in range(1,len(slope),1):
        
        #maxdiffSlop=max(slope)-min(slope)
        #maxdiffDiss=max(distence)-min(distence)
        #print("maxdiffSlop:"+str(maxdiffSlop))
        #print("maxdiffDiss:"+str(maxdiffDiss))
        MeanSlope=np.mean(slope)
        ratioSlope=slope/MeanSlope
        MeanDistance=np.mean(distance)
        ratioDistance=distance/MeanDistance
        
        distanceIdx=np.argsort(distance)
        sortIdx=np.argsort(slope) #smallest to largest
        #print(slope[sortIdx[len(sortIdx)//2]]-slope[sortIdx[len(sortIdx)//2-1]])
        #if (maxdiffSlop<((slope[sortIdx[len(sortIdx)//2]]-slope[sortIdx[len(sortIdx)//2-1]])*30)) and (maxdiffDiss<((distence[distenceIdx[len(distenceIdx)//2]]-distence[distenceIdx[len(distenceIdx)//2-1]])*100)) :
        
        #ipdb.set_trace()
        
        if (((max(ratioSlope)<max_threshold) and (min(ratioSlope)>min_threshold)) and ((max(ratioDistance)<max_threshold) and (min(ratioDistance)>min_threshold))) :
            #ipdb.set_trace()
            M=getMatchMatric(kp1, kp2,good)
            return M,good
        
        elif (max(ratioDistance)>max_threshold) or (min(ratioDistance)<min_threshold):
            leftDissDiff=distance[distanceIdx[len(distanceIdx)//2]]-distance[distanceIdx[0]]
            rightDissDiff=distance[distanceIdx[-1]]-distance[distanceIdx[len(distanceIdx)//2]]
            if (rightDissDiff>leftDissDiff):
                del good[distanceIdx[-1]]
                slope=np.delete(slope, distanceIdx[-1], 0)
                distance=np.delete(distance, distanceIdx[-1], 0)
            elif (rightDissDiff<leftDissDiff):
                del good[distanceIdx[0]]
                slope=np.delete(slope, distanceIdx[0], 0)
                distance=np.delete(distance, distanceIdx[0], 0)
                
        elif ((max(ratioSlope)>max_threshold) or (min(ratioSlope)<min_threshold)):
            leftSlopDiff=slope[sortIdx[len(sortIdx)//2]]-slope[sortIdx[0]]
            rightSlopDiff=slope[sortIdx[-1]]-slope[sortIdx[len(sortIdx)//2]]
            #print(sortIdx[-1])
            if rightSlopDiff>leftSlopDiff:
                del good[sortIdx[-1]]
                slope=np.delete(slope, sortIdx[-1], 0)
                distance=np.delete(distance, sortIdx[-1], 0)
            else:
                del good[sortIdx[0]]
                slope=np.delete(slope, sortIdx[0], 0)
                distance=np.delete(distance, sortIdx[0], 0)
   
    #ipdb.set_trace()
    M=getMatchMatric(kp1, kp2,good)
        
    return M,good


def check_alignment_result(img1,img2,M):
    global pp
    
    
    #need to know how the background map extend
    
    if M[0,1]<0: # moving up        
        imgTemp=np.zeros((int(-M[0,1]),img1.shape[1],3), dtype=np.uint8)
        img1=np.vstack((imgTemp,img1))  # img1 extend up
        anchorY=0
        if img2.shape[0]>img1.shape[0]:
            imgTemp=np.zeros((img2.shape[0]-img1.shape[0],img1.shape[1],3), dtype=np.uint8)
            img1=np.vstack((img1,imgTemp))
        
    if M[0,1]>=0:  # moving down
        if (img2.shape[0]+int(M[0,1])-img1.shape[0])>0:
            imgTemp=np.zeros((img2.shape[0]+int(M[0,1])-img1.shape[0],img1.shape[1],3), dtype=np.uint8)
            img1=np.vstack((img1,imgTemp))  # img1 extend down
        anchorY=int(M[0,1])
        
    if M[0,0]>=0: # moving right
        if (img2.shape[1]+int(M[0,0])-img1.shape[1])>0:
            imgTemp=np.zeros((img1.shape[0],img2.shape[1]+int(M[0,0])-img1.shape[1],3), dtype=np.uint8)
            img1=np.hstack((img1,imgTemp))  # img1 extend right
        anchorX=int(M[0,0])
        
    if M[0,0]<0:  # moving left
        imgTemp=np.zeros((img1.shape[0],int(-M[0,0]),3), dtype=np.uint8)
        img1=np.hstack((imgTemp,img1))  # img1 extend left
        anchorX=0
        if img2.shape[1]>img1.shape[1]:
            imgTemp=np.zeros((img1.shape[0],img2.shape[1]-img1.shape[1],3), dtype=np.uint8)
            img1=np.hstack((img1,imgTemp))
 


        
    dst_pad = np.zeros((img1.shape[0],img1.shape[1],3), dtype=np.uint8)
    dst_pad[anchorY:anchorY+img2.shape[0], anchorX:anchorX+img2.shape[1],:] = img2
    
    #ipdb.set_trace()
    overlapping = cv2.addWeighted(img1, 0.9, dst_pad, 0.9, 0)
    cv2.imwrite(path+"matching_result/matchingResult"+str(pp)+".jpg", overlapping)
    pp=pp+1

def read_M_file():
    # read the M
    f = open("matching M.txt","r") # w, r, a. if use r+, must read before write so that it can write after the original content. Or, it will replace!
    pos = f.tell()

    M_file=pd.DataFrame([[0,0,0,0]],columns=['img1','img2','M0','M1'])
    while True:        
        lines = f.readline() # read the whole line
        #print(lines)

        newpos = f.tell()
        if newpos == pos:  # stream position hasn't changed -> EOF
            f.close()
            #return 0
            break
        else:
            pos = newpos
                
        M=np.zeros((1,2))         
        _,img1,img2,M[0,0],M[0,1]= [i for i in lines.split()]
        
        M_file=pd.concat([M_file,pd.DataFrame([[img1,img2,M[0,0],M[0,1]]],columns=['img1','img2','M0','M1'])],axis=0)
    
    M_file=M_file.reset_index(drop=True)
    M_file=M_file.drop([0])
    
    return M_file

def rowNumAssignment():
    
    #raw_GPS=pd.read_csv('gps.csv')
    frame_GPS=pd.read_csv('frame_gps.csv')
    row_position=pd.read_csv('y2_cell.csv',header=None)

    
    # read the M
    M_file=read_M_file()
    
    
    # assign the row numbers to each frame based on the M
    row_num=np.zeros((len(row_position),16))
    row_152=np.zeros((len(row_position),16))
    row_152_flag_start=np.zeros((len(row_position),1))  # This relates to the last frame, but row_152_flag_start_2 relates to the current frame
    row_152_flag_end=np.zeros((len(row_position),1))
    for i in range(len(row_num)):
        #print(i)
        
        
         #ipdb.set_trace()
       
        row_spacing=frame_GPS.iloc[i,3]
        frame_name=row_position.iloc[i,0]
        row_numbers=row_position.iloc[i,1]
        row_points=row_position.iloc[i,2:row_numbers+2].reset_index(drop=True)
        
        if i==0:
            row_num[i,:row_numbers]=range(116, 116+row_numbers)
            row_152[i,:row_numbers]=row_points
            row_152_flag_start[0]=0
            row_152_flag_end[0]=row_numbers-1
            continue;
            
        M_index=M_file[M_file.iloc[:,1]==frame_name].index
        M=M_file.iloc[M_index[0]-1,2:4]
        img1=M_file.iloc[i-1,0]
        index_img1=row_position[row_position.iloc[:,0]==img1].index
        
        row_points2=row_points+M['M0']
        
        frame_row_start=False
        row_extend_flag=False
        row_152_flag_end_2=0
        row_152_2=np.zeros((1,16))
        
        # if i==46:
        #       break; 
        
        for ii in range(len(row_points)):
            #print(ii)
            
            if row_extend_flag==True: # new rows appear in the right of the current frame

                #if i==142:
                 #   ipdb.set_trace()
                for iii in range(1,10):
                    #print(iii)
                    
                    extend_row_position=row_152[index_img1,int(row_152_flag_end[index_img1,0])]+iii*frame_GPS.iloc[index_img1,3]
                    if np.abs(row_points2[ii]-extend_row_position.item())<(0.2*row_spacing):
                        row_num[i,ii]=row_num[index_img1,int(row_152_flag_end[index_img1,0])]+iii
                        row_152_2[0,ii]=row_points[ii]
                        row_152_flag_end_2=ii
                        break;
                continue;
            
            
            #ii=ii+1
            forward_rows_flag=False    
            for jj in [-1,-2,-3]:     # the currently frame is moving left and the left rows may not in the last frame
                #print(jj)
                if row_152[index_img1,0]!=0:
                    if (np.abs(row_points2[ii]-(row_152[index_img1,0]+jj*frame_GPS.iloc[index_img1,3]))<(0.2*row_spacing)).item() and (row_num[index_img1,0]+jj>0).item():
                        if row_num[index_img1,0]!=0:
                            row_num[i,ii]=row_num[index_img1,0]+jj
                            row_152_2[0,ii]=row_points[ii]
                        else:
                            for iii in range(1,4):
                                if row_num[index_img1,0+iii]!=0:
                                    row_num[i,ii]=row_num[index_img1,jj+iii]-iii
                                    row_152_2[0,ii]=row_points[ii]
                                    break;
                        row_152_flag_start_2=ii #The row number in this frame is starting in #jj 
                        frame_row_start=True
                        forward_rows_flag=True
                        break;
                        
                       
            
            for jj in range(int(row_152_flag_start[index_img1,0].item()),int(row_152_flag_end[index_img1,0].item()+1)): # match with the rows appeared in the last frame
                #print(jj)
                
                if forward_rows_flag==True:
                    break;
                    
                if np.abs(row_points2[ii]-row_152[index_img1,jj])<(0.2*row_spacing):
                    if row_num[index_img1,jj]!=0:
                        row_num[i,ii]=row_num[index_img1,jj]
                    else:
                        for iii in range(1,4):
                            if row_num[index_img1,jj+iii]!=0:
                                row_num[i,ii]=row_num[index_img1,jj+iii]-iii
                                break;
                    row_152_2[0,ii]=row_points[ii]
                    row_152_flag_end_2=ii
                    #print(row_152_flag_end_2)
                    
                    
                    if row_num[i,ii]==row_num[index_img1,int(row_152_flag_end[index_img1,0])]:
                        row_extend_flag=True  # new rows appear
                        break;
                    
                    if frame_row_start==False:
                        row_152_flag_start_2=ii #The row number in this frame is starting in #jj 
                        frame_row_start=True 
                        
                    break;
                    
            
        #ipdb.set_trace()
        
        for ii in range(len(row_points)):  # some rows may not be detected in the last frame
            #print(ii)
            
            scan_forward_flag=False
            scan_backward_flag=False
            if ii==0:
                scan_forward_flag=True
                scan_backward_flag=False
                   
            elif (ii==len(row_points)-1):
                scan_forward_flag=False
                scan_backward_flag=True
            
            else:
                scan_forward_flag=True
                scan_backward_flag=True
                
            value_change_flag=False
            if scan_forward_flag==True:
                if np.abs(row_points[ii+1]-row_points[ii]-row_spacing)<(0.2*row_spacing):
                    if row_num[i,ii+1]!=0:
                        row_num[i,ii]=row_num[i,ii+1]-1
                        row_152_2[0,ii]=row_points[ii]
                        value_change_flag=True
                    else:
                        for iii in range(1,4):
                            if (ii+1+iii<len(row_points)) and row_num[i,ii+1+iii]!=0:
                                row_num[i,ii]=row_num[i,ii+1+iii]-iii-1
                                row_152_2[0,ii]=row_points[ii]
                                value_change_flag=True
                                break;
                    
                    

            if value_change_flag==False and scan_backward_flag==True:
                if np.abs(row_points[ii]-row_points[ii-1]-row_spacing)<(0.2*row_spacing):
                    if row_num[i,ii-1]!=0:
                        row_num[i,ii]=row_num[i,ii-1]+1
                        row_152_2[0,ii]=row_points[ii]
                        value_change_flag=True
                    else:
                        for iii in range(1,4):
                            if (ii-1-iii>=0) and (row_num[i,ii-1-iii]!=0):
                                if np.abs(row_points[ii]-row_points[ii-1-iii]-(iii+1)*row_spacing)<(0.2*row_spacing):
                                    row_num[i,ii]=row_num[i,ii-1-iii]+iii+1
                                    row_152_2[0,ii]=row_points[ii]
                                    value_change_flag=True
                                    break;
     
            if value_change_flag==True:          
                if row_152_flag_start_2>ii:
                    row_152_flag_start_2=ii
                if row_152_flag_end_2<ii:
                    row_152_flag_end_2=ii
                        
            
                
        temp_row_start=row_num[i,row_152_flag_start_2]
        while (row_152_flag_start_2-1>=0) and (temp_row_start>1):
            row_152_2[0,row_152_flag_start_2-1]=row_152_2[0,row_152_flag_start_2]-row_spacing
            row_152_flag_start_2=row_152_flag_start_2-1
            temp_row_start=temp_row_start-1
                
        row_152_flag_start[i]=row_152_flag_start_2
        row_152_flag_end[i]=row_152_flag_end_2
        row_152[i,:]=row_152_2
        #print(row_152_flag_start,row_152_flag_end)
        
    
    return 0

    
def Begining_End_cropRows():
    
    frame_GPS=pd.read_csv('frame_gps.csv')
    row_spacing_ground=0.97
    #row_position=pd.read_csv('y2_cell.csv',header=None)
    #img_size=pd.read_csv('img_size.csv',header=None)
    
    # read the M
    M_file=read_M_file()
    
    frame_GPS['row_pixel']=np.zeros((len(frame_GPS),1))
    frame_GPS['sum_row_pixel']=np.zeros((len(frame_GPS),1))
   
    for i in range(len(M_file)):
        img2=M_file.iloc[i,1]
        frame_GPS.iloc[i+1,5]=float(M_file.iloc[i,3])/((frame_GPS[frame_GPS.iloc[:,2]==img2]['row_spacing']).iloc[0]/row_spacing_ground)
    
    frame_GPS.iloc[0,6]=0    # 221.4378 83.992-1605/362 68.4971+1806/391
    for i in range(len(M_file)):            
        img1=M_file.iloc[i,0]
        img2=M_file.iloc[i,1]
        index_img1=frame_GPS[frame_GPS.iloc[:,2]==img1].index
        index_img2=frame_GPS[frame_GPS.iloc[:,2]==img2].index
        frame_GPS.iloc[i+1,6]=np.array(frame_GPS.iloc[index_img1,6])+np.array(frame_GPS.iloc[index_img2,5])

    frame_GPS.to_csv(path+'frame_GPS_beginning_end.csv', index=False)


def img_connection_with_row_assignment_and_Begining_End():
    frame_GPS=pd.read_csv('frame_gps.csv')
    row_assignment=pd.read_csv('row_assignment.csv')
    imgs_meter=pd.read_csv('gps.csv')
    imgs_size=pd.read_csv('img_size.csv')
    
    results=pd.DataFrame([],columns=imgs_meter.columns)
    results['row_assigned']=np.array([])
    results['begining_end']=np.array([])
    
    for i in range(len(frame_GPS)):
        img=frame_GPS.iloc[i,2]
        row_spacing=frame_GPS.iloc[i,3]
        row_num=frame_GPS.iloc[i,4]
        sum_row_pixel=frame_GPS.iloc[i,6]
        frame_size=imgs_size.iloc[i,1:]
        row_assigned=row_assignment.iloc[i,1:]
        
        center_x_meter=sum_row_pixel+(frame_size[0]/2)/row_spacing
        
        #if i<72:
        #    j_start=0
        #else:
        j_start=1
        for j in range(j_start,row_num):  # will skip the row in the edge of a frame
            #print(j)
            
            if row_assigned[j]==0:
                continue;
            
            
            temp=imgs_meter[(imgs_meter.iloc[:,3]==img) & (imgs_meter.iloc[:,5]==j+1)]
            temp['row_assigned']=row_assigned[j]
            temp['begining_end']=center_x_meter+temp.iloc[:,4]
            
            results=pd.concat([results,temp])
            del temp


    results.to_csv('img_meters.csv',index=False)
    
    
def map_315_152():
    seedlings=pd.read_csv('img_meters.csv')
    map=np.zeros((315,152))
    
    
    for i in range(152):
        #print(i)
        
        temp=seedlings[(i<seedlings['row_assigned']) & (seedlings['row_assigned']<=((i+1)*1))]
        
        begining=min(temp['begining_end'])
        end=max(temp['begining_end'])
        middle=(end-begining)/315
        
        for j in range(315):
            value=temp[(j*middle<temp['begining_end']) & (temp['begining_end']<=((j+1)*middle))]
            if len(value)!=0:
                map[j,i]=np.mean(value.iloc[:,10])

    plt.imshow(map,vmin=10,vmax=35,cmap=plt.cm.jet)
    plt.axis('off')
    plt.colorbar()
    
def map_63_38():
    map_SC = np.load('./map_315_152_SC.npy')
    map_CZ=np.load('./map_315_152_CZ.npy')
    
    map_SC[map_SC == 0] = np.nan
    map_CZ[map_CZ == 0] = np.nan
    
    map_SC_63_38=np.zeros((63,38))
    map_CZ_63_38=np.zeros((63,38))
    
    for i in range(38):
        tenp1=map_SC [:,i*4:(i+1)*4]
        tenp2=map_CZ[:,i*4:(i+1)*4]
        
        for j in range(63):
            
            map_SC_63_38[j,i]=np.nanmean(tenp1[j*5:(j+1)*5,:])
            map_CZ_63_38[j,i]=np.nanmean(tenp2[j*5:(j+1)*5,:])
            
    plt.imshow(map_SC_63_38,vmin=5,vmax=18,cmap=plt.cm.jet)
    plt.axis('off')
    plt.colorbar()
    
    plt.imshow(map_CZ_63_38,vmin=10,vmax=35,cmap=plt.cm.jet)
    plt.axis('off')
    plt.colorbar()
    
    np.save('map_SC_63_38.npy', map_SC_63_38)
    np.save('map_CZ_63_38.npy', map_CZ_63_38)
        
    map_SC_63_38=np.load('./map_SC_63_38.npy')
    map_CZ_63_38=np.load('./map_CZ_63_38.npy')
    
    temp1=np.reshape(map_SC_63_38,(-1,1))
    temp2=np.reshape(map_CZ_63_38,(-1,1))
    
    temp3=np.reshape(np.array(soil['Elevation']),(63,38))
    plt.imshow(temp3,cmap=plt.cm.jet)
    plt.axis('off')
    plt.colorbar()
    

f = open("matching_M.txt","a") # w, r, a. if use r+, must read before write so that it can write after the original content. Or, it will replace!
img1=cv2.imread(filename[-1])
img1=cv2.imread(filename[0])
for i in range(1,len(filename)-1,1):

    #print(i)
    img2=cv2.imread(filename[i])
    kp1,kp2,kp_pairs,M=match_images(img1, img2,ratio=0.6,feature_threshold=400,feature_filter=3,feature_movement=1,remove_bad=1)
    
    drawMyMatches(img1,kp1,img2,kp2,kp_pairs,flag=0)
    check_alignment_result(img1,img2,M)

    filename[0].split('\\')[1]
    f.write(str(i)+" "+filename[i-1].split('\\')[1]+"  "+filename[i].split('\\')[1]+" "+str(M[0,0])+" "+str(M[0,1])+"\n")

    img1=img2
    
f.close()


'''
pp=0
f = open(path+"matching M.txt","a") # w, r, a. if use r+, must read before write so that it can write after the original content. Or, it will replace!

#fix=[141,214,283,356,425,494]

print("start: " ,time.asctime(time.localtime(time.time())))
start=time.time()

for i in range(0,len(filename)-1):

    #print(i)
   
    img1=cv2.imread(filename[i])
    img2=cv2.imread(filename[i+1])
    kp1,kp2,kp_pairs,M=match_images(img1, img2,ratio=0.6,feature_threshold=400,feature_filter=3,feature_movement=1,remove_bad=1)
   
    
    
    src_pts = np.float32([ kp1[m[0].queryIdx].pt for m in kp_pairs ])
    dst_pts = np.float32([ kp2[m[0].trainIdx].pt for m in kp_pairs ])
    
    print(src_pts[148])
    print(dst_pts[148])
    
    M[0,0]=-178
    M[0,1]=-3370
    
    
    
    drawMyMatches(img1,kp1,img2,kp2,kp_pairs,flag=0)
    check_alignment_result(img1,img2,M)
    
    #---------------------------------------------------------------------

    filename[0].split('\\')[1]
    f.write(str(i)+" "+filename[i].split('\\')[1]+"  "+filename[i+1].split('\\')[1]+" "+str(M[0,0])+" "+str(M[0,1])+"\n")
    
print("end: " ,time.asctime(time.localtime(time.time())))
end=time.time()
print("used time (s): " ,end-start)
    
f.close()
'''