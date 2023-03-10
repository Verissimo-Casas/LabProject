import torch
import cv2
from datetime import datetime
import os
import shutil
import numpy as np
import json

import bank
import alerts

f = open('parameters.json')
parameters = json.load(f)

scale = parameters['escala']
mac = parameters['mac']
min_tempoEPI = parameters['min_tempoEPI']

model = torch.hub.load('./yolov5', 'custom', path='./resources/best.pt', source='local')
model.conf = parameters['confianca']
model.iou = parameters['iou']
names = ['capacete', 'colete', 'gancho', 'luva', 'oculos', 'pessoa']
names_color = [(0,0,255), (0,255,0), (255,0,0), (255,255,0), (255,0,255), (0,255,255)]

def predict(frame):
    #img = cv2.resize(img, (640,640))
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(img)
    res = results.xyxy[0]

    ratio_x = frame.shape[1]/img.shape[1]
    ratio_y = frame.shape[0]/img.shape[0]
    bb_list = []
    for rect in res:
        x1 = int(ratio_x*rect[0]/scale)
        y1 = int(ratio_y*rect[1]/scale)
        x2 = int(ratio_x*rect[2]/scale)
        y2 = int(ratio_y*rect[3]/scale)
        conf = float(rect[4])
        label = int(rect[5])
        bb_list.append( {'xmin': x1, 'ymin' : y1, 'xmax' : x2, 'ymax' : y2, 'label': label, 'confidence':conf}  )
    
    return bb_list

def draw(frame,bb_list):
    for bb in bb_list:
        name = names[bb['label']]
        color = names_color[bb['label']]
        conf = bb['confidence']
 
        cv2.rectangle(frame, (int(bb['xmin']*scale), int(bb['ymin']*scale)), (int(bb['xmax']*scale), int(bb['ymax']*scale)), color,2)
        cv2.rectangle(frame, (int(bb['xmin']*scale), int(-20 + bb['ymin']*scale)), (int(bb['xmax']*scale), int(bb['ymin']*scale)), color, -1)
        cv2.putText(frame, '%s' % (name), (int(bb['xmin']*scale), int(-5+bb['ymin']*scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2, cv2.LINE_AA)
            
    return frame

def transform(image):
    #image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    #image_h, image_w = image.shape[1], image.shape[0]
    #image = cv2.resize(image, (518, 921))
    #image_h, image_w = image.shape[0], image.shape[1]
    return image


helmet_start = False
vest_start = False
glasses_start = False
gloves_start = False

time_helmet_start = datetime.now()
time_vest_start = datetime.now()
time_gloves_start = datetime.now()
time_glasses_start = datetime.now()

time_helmet_delta = 0.0
time_vest_delta = 0.0
time_gloves_delta = 0.0
time_glasses_delta = 0.0

frame_counter_helmet = 0
frame_counter_vest = 0
frame_counter_gloves = 0
frame_counter_glasses = 0

events_folder = './data/eventos/None'
events_folder_helmet = './data/eventos/None'
events_folder_vest = './data/eventos/None'
events_folder_gloves = './data/eventos/None'
events_folder_glasses = './data/eventos/None'

def checkEPI(frame, bb_list):
    global helmet_start, vest_start, gloves_start, glasses_start
    global time_helmet_start, time_vest_start, time_gloves_start, time_glasses_start
    global time_helmet_delta, time_vest_delta, time_gloves_delta, time_glasses_delta
    global frame_counter_helmet, frame_counter_vest, frame_counter_gloves, frame_counter_glasses
    global events_folder, events_folder_helmet, events_folder_vest, events_folder_gloves, events_folder_glasses

    pessoas = []
    capacetes = []
    coletes = []
    luvas = []
    oculos = []
    for bb in bb_list:
        if bb['label'] == 5:
            pessoas.append(bb)
        if bb['label'] == 0:
            capacetes.append(bb)
        if bb['label'] == 1:
            coletes.append(bb)
        if bb['label'] == 3:
            luvas.append(bb)
        if bb['label'] == 4:
            oculos.append(bb)

    pessoas_epi = [[False, False, False, False] for _ in range(len(pessoas))]
    for k in range(len(pessoas)):
        pessoa = pessoas[k]
        xmin = pessoa['xmin']
        ymin = pessoa['ymin']
        xmax = pessoa['xmax']
        ymax = pessoa['ymax']

        for capacete in capacetes:
            xc = 0.5*(capacete['xmin'] + capacete['xmax'])
            yc = 0.5*(capacete['ymin'] + capacete['ymax'])
            if (xmin < xc < xmax) and (ymin < yc < ymax):
                pessoas_epi[k][0] = True

        for colete in coletes:
            xc = 0.5*(colete['xmin'] + colete['xmax'])
            yc = 0.5*(colete['ymin'] + colete['ymax'])
            if (xmin < xc < xmax) and (ymin < yc < ymax):
                pessoas_epi[k][1] = True
        
        '''
        for luva in luvas:
            xc = 0.5*(luva['xmin'] + luva['xmax'])
            yc = 0.5*(luva['ymin'] + luva['ymax'])
            if (xmin < xc < xmax) and (ymin < yc < ymax):
                pessoas_epi[k][2] = True
        
        for ocl in oculos:
            xc = 0.5*(ocl['xmin'] + ocl['xmax'])
            yc = 0.5*(ocl['ymin'] + ocl['ymax'])
            if (xmin < xc < xmax) and (ymin < yc < ymax):
                pessoas_epi[k][3] = True
        '''
        
    if len(pessoas) == 0:
        helmet_start = False
        vest_start = False
        gloves_start = False
        glasses_start = False

    for k in range(len(pessoas)):
        pessoa = pessoas[k]
        xmin = int(scale*pessoa['xmin'])
        ymin = int(scale*pessoa['ymin'])
        xmax = int(scale*pessoa['xmax'])
        ymax = int(scale*pessoa['ymax'])

        #helmet detector
        if pessoas_epi[k][0] == False:
            if helmet_start == False:
                time_helmet_start = datetime.now()
                helmet_start = True
                events_folder_helmet = './data/eventos/capacete_%s' % time_helmet_start
                os.mkdir(events_folder_helmet)
        else:
            helmet_start = False

        #vest detector
        if pessoas_epi[k][1] == False:
            if vest_start == False:
                time_vest_start = datetime.now()
                vest_start = True
                events_folder_vest = './data/eventos/colete_%s' % time_vest_start
                os.mkdir(events_folder_vest)
        else:
            vest_start = False
        
        '''
        #gloves detector
        if pessoas_epi[k][2] == False:
            if gloves_start == False:
                time_gloves_start = datetime.now()
                gloves_start = True
                events_folder_gloves = './data/eventos/luvas_%s' % time_gloves_start
                os.mkdir(events_folder_gloves)
        else:
            gloves_start = False
    
        #glasses detector
        if pessoas_epi[k][3] == False:
            if glasses_start == False:
                time_glasses_start = datetime.now()
                glasses_start = True
                events_folder_glasses = './data/eventos/oculos_%s' % time_glasses_start
                os.mkdir(events_folder_glasses)
        else:
            glasses_start = False
        '''
        
    if helmet_start == True:
        time_helmet_finish = datetime.now()
        time_helmet_delta = (time_helmet_finish - time_helmet_start).total_seconds()
        cv2.imwrite(events_folder_helmet + '/%s.jpg' % time_helmet_finish, frame)
    else:
        if time_helmet_delta >= min_tempoEPI:
            data = [[mac,datetime.now(),time_helmet_delta,events_folder_helmet,1]] ################# Error de tipeo
            alerts.sendEmail(time_helmet_delta, events_folder_helmet, 1)
            bank.sendData(data)
        elif time_helmet_delta < min_tempoEPI and time_helmet_delta != 0.0:
            if os.path.isdir(events_folder_helmet) == True:
                shutil.rmtree(events_folder_helmet)
        time_helmet_delta = 0.0

    if vest_start == True:
        time_vest_finish = datetime.now()
        time_vest_delta = (time_vest_finish - time_vest_start).total_seconds()
        cv2.imwrite(events_folder_vest + '/%s.jpg' % time_vest_finish, frame)
    else:
        if time_vest_delta >= min_tempoEPI:
            data = [[mac,datetime.now(),time_vest_delta,events_folder_vest,2]]
            alerts.sendEmail(time_vest_delta, events_folder_vest, 2)
            bank.sendData(data)
        elif time_vest_delta < min_tempoEPI and time_vest_delta != 0.0:
            if os.path.isdir(events_folder_vest) == True:
                shutil.rmtree(events_folder_vest)
        time_vest_delta = 0.0
    
    '''
    if gloves_start == True:
        time_gloves_finish = datetime.now()
        time_gloves_delta = (time_gloves_finish - time_gloves_start).total_seconds()
        cv2.imwrite(events_folder_gloves + '/%s.jpg' % time_gloves_finish, frame)
    else:
        if time_gloves_delta >= min_tempoEPI:
            data = [[mac,datetime.now(),time_gloves_delta,events_folder_gloves,3]]
            alerts.sendEmail(time_gloves_delta, events_folder_gloves,3)
            bank.sendData(data)
        elif time_gloves_delta < min_tempoEPI and time_gloves_delta != 0.0:
            if os.path.isdir(events_folder_gloves) == True:
                shutil.rmtree(events_folder_gloves)
        time_gloves_delta = 0.0
        
    if glasses_start == True:
        time_glasses_finish = datetime.now()
        time_glasses_delta = (time_glasses_finish - time_glasses_start).total_seconds()
        cv2.imwrite(events_folder_glasses + '/%s.jpg' % time_glasses_finish, frame)
    else:
        if time_glasses_delta >= min_tempoEPI:
            data = [[mac,datetime.now(),time_glasses_delta,events_folder_glasses,4]]
            alerts.sendEmail(time_glasses_delta, events_folder_glasses, 4)
            bank.sendData(data)
        elif time_glasses_delta < min_tempoEPI and time_glasses_delta != 0.0:
            if os.path.isdir(events_folder_glasses) == True:
                shutil.rmtree(events_folder_glasses)
        time_glasses_delta = 0.0
    '''
    
icon_helmet = cv2.imread('./icons/capacete.png')
icon_helmet = 255 - cv2.resize(icon_helmet, (100,100))

icon_vest = cv2.imread('./icons/colete.png')
icon_vest = 255 - cv2.resize(icon_vest, (100,100))

icon_gloves = cv2.imread('./icons/luvas.png')
icon_gloves = 255 - cv2.resize(icon_gloves, (100,100))

icon_glasses = cv2.imread('./icons/oculos.png')
icon_glasses = 255 - cv2.resize(icon_glasses, (100,100))

def draw_icons(frame, bb_list):
    img = cv2.copyMakeBorder(frame, 500, 500, 500, 500, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    pessoas = []
    coletes = []
    capacetes = []
    oculos = []
    luvas = []
    for bb in bb_list:
        if bb['label'] == 5:
            pessoas.append(bb)
        if bb['label'] == 0:
            capacetes.append(bb)
        if bb['label'] == 1:
            coletes.append(bb)
        if bb['label'] == 4:
            oculos.append(bb)
        if bb['label'] == 3:
            luvas.append(bb)

    pessoas_epi = [[False, False, False, False] for _ in range(len(pessoas))]
    for k in range(len(pessoas)):
        pessoa = pessoas[k]
        xmin = pessoa['xmin']
        ymin = pessoa['ymin']
        xmax = pessoa['xmax']
        ymax = pessoa['ymax']

        for capacete in capacetes:
            xc = 0.5*(capacete['xmin'] + capacete['xmax'])
            yc = 0.5*(capacete['ymin'] + capacete['ymax'])
            if (xmin < xc < xmax) and (ymin < yc < ymax):
                pessoas_epi[k][0] = True

        for colete in coletes:
            xc = 0.5*(colete['xmin'] + colete['xmax'])
            yc = 0.5*(colete['ymin'] + colete['ymax'])
            if (xmin < xc < xmax) and (ymin < yc < ymax):
                pessoas_epi[k][1] = True
        
        for luva in luvas:
            xc = 0.5*(luva['xmin'] + luva['xmax'])
            yc = 0.5*(luva['ymin'] + luva['ymax'])
            if (xmin < xc < xmax) and (ymin < yc < ymax):
                pessoas_epi[k][2] = True

        for glasses in oculos:
            xc = 0.5*(glasses['xmin'] + glasses['xmax'])
            yc = 0.5*(glasses['ymin'] + glasses['ymax'])
            if (xmin < xc < xmax) and (ymin < yc < ymax):
                pessoas_epi[k][3] = True

    H, W = img.shape[0], img.shape[1]
    N = len(pessoas)
    helmet_off = 0
    vest_off = 0
    gloves_off = 0
    glasses_off = 0
    for k in range(len(pessoas)):
        pessoa = pessoas[k]
        xmin = int(scale*pessoa['xmin']) + 500
        ymin = int(scale*pessoa['ymin']) + 500
        xmax = int(scale*pessoa['xmax']) + 500
        ymax = int(scale*pessoa['ymax']) + 500
        length_scale = ((xmax-xmin)/W)**0.5
        cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (0,0,255), 2)

        #helmet detector
        if pessoas_epi[k][0] == False:
            icon_helmet_copy = 1*icon_helmet
            icon_helmet_copy = cv2.resize(icon_helmet_copy, (int(100*length_scale),int(100*length_scale)))
            h,w = icon_helmet_copy.shape[0], icon_helmet_copy.shape[1]
            cx = int(0.5*(2*xmin + w))
            cy = int(0.5*(2*ymin - h))
            cv2.circle(img, (cx,cy), int(0.75*h), (0,0,255), -1)
            img[ymin-h:ymin,xmin:xmin+w] = np.where(icon_helmet_copy > 200, icon_helmet_copy ,img[ymin-h:ymin,xmin:xmin+w])
            helmet_off += 1
        else:
            icon_helmet_copy = 1*icon_helmet
            icon_helmet_copy = cv2.resize(icon_helmet_copy, (int(100*length_scale),int(100*length_scale)))
            h,w = icon_helmet_copy.shape[0], icon_helmet_copy.shape[1]
            cx = int(0.5*(2*xmin + w))
            cy = int(0.5*(2*ymin - h))
            cv2.circle(img, (cx,cy), int(0.75*h), (0,255,0), -1)
            img[ymin-h:ymin,xmin:xmin+w] = np.where(icon_helmet_copy > 200, icon_helmet_copy ,img[ymin-h:ymin,xmin:xmin+w])
        
        #vest detector
        if pessoas_epi[k][1] == False:
            icon_vest_copy = 1*icon_vest
            icon_vest_copy = cv2.resize(icon_vest_copy, (int(100*length_scale),int(100*length_scale)))
            h,w = icon_vest_copy.shape[0], icon_vest_copy.shape[1]
            cx = int(1.5*w + 0.5*(2*xmin + w))
            cy = int(0.5*(2*ymin - h))
            cv2.circle(img, (cx,cy), int(0.75*h), (0,0,255), -1)
            img[ymin-h:ymin,xmin+w+int(0.5*w):xmin+int(2.5*w)] = np.where(icon_vest_copy > 200, icon_vest_copy ,img[ymin-h:ymin,xmin+w+int(0.5*w):xmin+int(2.5*w)])
            vest_off += 1
        else:
            icon_vest_copy = 1*icon_vest
            icon_vest_copy = cv2.resize(icon_vest_copy, (int(100*length_scale),int(100*length_scale)))
            h,w = icon_helmet_copy.shape[0], icon_helmet_copy.shape[1]
            cx = int(1.5*w + 0.5*(2*xmin + w))
            cy = int(0.5*(2*ymin - h))
            cv2.circle(img, (cx,cy), int(0.75*h), (0,255,0), -1)
            img[ymin-h:ymin,xmin+w+int(0.5*w):xmin+int(2.5*w)] = np.where(icon_vest_copy > 200, icon_vest_copy ,img[ymin-h:ymin,xmin+w+int(0.5*w):xmin+int(2.5*w)])
        
        
        #gloves
        icon_gloves_copy = 1*icon_gloves
        icon_gloves_copy = cv2.resize(icon_gloves_copy, (int(100*length_scale),int(100*length_scale)))
        h,w = icon_gloves_copy.shape[0], icon_gloves_copy.shape[1]
        cx = int(3*w + 0.5*(2*xmin + w))
        cy = int(0.5*(2*ymin - h))
        cv2.circle(img, (cx,cy), int(0.75*h), (150,150,150), -1)
        img[ymin-h:ymin,xmin+w+int(2*w):xmin+int(4*w)] = np.where(icon_gloves_copy > 250, icon_gloves_copy ,img[ymin-h:ymin,xmin+w+int(2*w):xmin+int(4*w)])
        
        #glasses
        icon_glasses_copy = 1*icon_glasses
        icon_glasses_copy = cv2.resize(icon_glasses_copy, (int(100*length_scale),int(100*length_scale)))
        h,w = icon_glasses_copy.shape[0], icon_glasses_copy.shape[1]
        cx = int(4.5*w + 0.5*(2*xmin + w))
        cy = int(0.5*(2*ymin - h))
        cv2.circle(img, (cx,cy), int(0.75*h), (150,150,150), -1)
        img[ymin-h:ymin,xmin+w+int(3.5*w):xmin+int(5.5*w)] = np.where(icon_glasses_copy > 250, icon_glasses_copy ,img[ymin-h:ymin,xmin+w+int(3.5*w):xmin+int(5.5*w)])
        
        '''
        #gloves detector
        if pessoas_epi[k][2] == False:
            gloves_off += 1
            icon_gloves_copy = 1*icon_gloves
            icon_gloves_copy = cv2.resize(icon_gloves_copy, (int(100*length_scale),int(100*length_scale)))
            h,w = icon_gloves_copy.shape[0], icon_gloves_copy.shape[1]
            cx = int(3*w + 0.5*(2*xmin + w))
            cy = int(0.5*(2*ymin - h))
            cv2.circle(img, (cx,cy), int(0.75*h), (0,0,255), -1)
            img[ymin-h:ymin,xmin+w+int(2*w):xmin+int(4*w)] = np.where(icon_gloves_copy > 250, icon_gloves_copy ,img[ymin-h:ymin,xmin+w+int(2*w):xmin+int(4*w)])
        else:
            icon_gloves_copy = 1*icon_gloves
            icon_gloves_copy = cv2.resize(icon_gloves_copy, (int(100*length_scale),int(100*length_scale)))
            h,w = icon_gloves_copy.shape[0], icon_gloves_copy.shape[1]
            cx = int(3*w + 0.5*(2*xmin + w))
            cy = int(0.5*(2*ymin - h))
            cv2.circle(img, (cx,cy), int(0.75*h), (0,255,0), -1)
            img[ymin-h:ymin,xmin+w+int(2*w):xmin+int(4*w)] = np.where(icon_gloves_copy > 250, icon_gloves_copy ,img[ymin-h:ymin,xmin+w+int(2*w):xmin+int(4*w)])

        #glasses detector
        if pessoas_epi[k][3] == False:
            glasses_off += 1
            icon_glasses_copy = 1*icon_glasses
            icon_glasses_copy = cv2.resize(icon_glasses_copy, (int(100*length_scale),int(100*length_scale)))
            h,w = icon_glasses_copy.shape[0], icon_glasses_copy.shape[1]
            cx = int(4.5*w + 0.5*(2*xmin + w))
            cy = int(0.5*(2*ymin - h))
            cv2.circle(img, (cx,cy), int(0.75*h), (0,0,255), -1)
            img[ymin-h:ymin,xmin+w+int(3.5*w):xmin+int(5.5*w)] = np.where(icon_glasses_copy > 250, icon_glasses_copy ,img[ymin-h:ymin,xmin+w+int(3.5*w):xmin+int(5.5*w)])
        else:
            icon_glasses_copy = 1*icon_glasses
            icon_glasses_copy = cv2.resize(icon_glasses_copy, (int(100*length_scale),int(100*length_scale)))
            h,w = icon_glasses_copy.shape[0], icon_glasses_copy.shape[1]
            cx = int(4.5*w + 0.5*(2*xmin + w))
            cy = int(0.5*(2*ymin - h))
            cv2.circle(img, (cx,cy), int(0.75*h), (0,255,0), -1)
            img[ymin-h:ymin,xmin+w+int(3.5*w):xmin+int(5.5*w)] = np.where(icon_glasses_copy > 250, icon_glasses_copy ,img[ymin-h:ymin,xmin+w+int(3.5*w):xmin+int(5.5*w)])
        '''
    H,W = img.shape[0], img.shape[1]
    img = img[500:H-500,500:W-500]
    
    return img
