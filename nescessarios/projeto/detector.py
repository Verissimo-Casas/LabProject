import torch
import cv2
from datetime import datetime
import os
import shutil
import numpy as np

import bank
import alerts


mac = "00:E1:8C:70:F4:6D"
scale = 0.8
min_tempoCELL = 2


model = torch.hub.load('./yolov5', 'custom', path='yolov5s.pt', source='local')
# model.conf = 0.4132075381278992
# model.cuda()
names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', ' fire hydrant', ' stop sign', ' parking meter', ' bench', ' bird', ' cat', ' dog', ' horse', ' sheep', ' cow', ' elephant', ' bear', ' zebra', ' giraffe', ' backpack', ' umbrella', ' handbag', ' tie', ' suitcase', ' frisbee', ' skis', ' snowboard', ' sports ball', ' kite', ' baseball bat', ' baseball glove', ' skateboard', ' surfboard', ' tennis racket', ' bottle', ' wine glass', ' cup', ' fork', ' knife', ' spoon', ' bowl', ' banana', ' apple', ' sandwich', ' orange', ' broccoli', ' carrot', ' hot dog', ' pizza', ' donut', ' cake', ' chair', ' couch', ' potted plant', ' bed', ' dining table', ' toilet', ' tv', ' laptop', ' mouse', ' remote', ' keyboard', ' cell phone', ' microwave', ' oven', ' toaster', ' sink', ' refrigerator', ' book', ' clock', ' vase', ' scissors', ' teddy bear', ' hair drier', ' toothbrush']
names_color = []
for i in range(len(names)):
    colores = tuple(np.random.randint(0, 255, 3).tolist())
    names_color.append(colores)

def detect(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model(img)
    results = results.xyxy[0]

    ratio = img.shape[2]/img.shape[2]
    bb_list = []
    for index in results:
        x1 = int(ratio * index[0]/scale)
        y1 = int(ratio * index[1]/scale)
        x2 = int(ratio * index[2]/scale)
        y2 = int(ratio * index[3]/scale)
        label = int(index[5])
        conf = float(index[4])
        bb_list.append({
            'xmin': x1,
            'ymin': y1,
            'xmax': x2,
            'ymax': y2,
            'label': label,
            'confidence': conf
        })
    return bb_list



def draw(img, bb_list):
    for bb in bb_list:
        if bb['label'] == 0 or bb['label'] == 28 or bb['label'] == 67:
            name = names[bb['label']]
            color = names_color[bb['label']]
            x1, y1 = int(bb['xmin'] * scale), int(bb['ymin'] * scale)
            x2, y2 = int(bb['xmax'] * scale), int(bb['ymax'] * scale)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
            cv2.rectangle(img, (x1, y1-15), (x2, y1), color, -1)
            cv2.putText(img, name, (x1, y1-2), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1, cv2.LINE_AA)
    return img

def area_pts(frame): 
    cv2.rectangle(frame,(0,0),(frame.shape[1],40),(0,0,0),-1)
    color = (0, 255, 0)
    texto_estado = "Estado: Tudo bom!"
    cv2.putText(frame, texto_estado , (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX, 1, color,2)

cell_start = False
time_cell_start = datetime.now()
time_cell_delta = 0.0
frame_counter_cell = 0

events_folder_cell = './data/eventos'

def checkCELL(frame, bb_list):
    global cell_start
    global time_cell_start
    global time_cell_delta
    global frame_counter_cell
    global events_folder_cell

    pessoas = []
    telefones = []

    for bb in bb_list:
        if bb['label'] == 0:
            pessoas.append(bb)
        if bb['label'] == 67:
            telefones.append(bb)

    pessoas_cell = [[True] for _ in range(len(pessoas))]
    for k in range(len(pessoas)):
        pessoa = pessoas[k]
        xmin = pessoa['xmin']
        ymin = pessoa['ymin']
        xmax = pessoa['xmax']
        ymax = pessoa['ymax']
        #print(k, ': personas')

        for telefone in telefones: # 
            xc = 0.5*(telefone['xmin'] + telefone['xmax'])
            yc = 0.5*(telefone['ymin'] + telefone['ymax'])
            if (xmin < xc < xmax) and (ymin < yc < ymax):
                pessoas_cell[k][0] = True
            #print(pessoas_cell[k][0])
            #print(telefone)
            #print('xc: ', xc)
            #print('yc: ', yc)
            cv2.rectangle(frame,(0,0),(frame.shape[1],40),(0,0,0),-1)
            color = (0, 0, 255)
            texto_estado = "Estado: telefone detetado!"
            cv2.putText(frame, texto_estado , (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, color,2)

                        
    if len(pessoas) == 0:
        cell_start = False

    for k in range(len(pessoas)):
        pessoa = pessoas[k]
        xmin = int(scale*pessoa['xmin'])
        ymin = int(scale*pessoa['ymin'])
        xmax = int(scale*pessoa['xmax'])
        ymax = int(scale*pessoa['ymax'])

        #cell detector
        if pessoas_cell[k][0] == False:
            if cell_start == False:
                time_cell_start = datetime.now()
                cell_start = True
                events_folder_cell = './data/eventos/telefone_%s' % time_cell_start
        else:
            cell_start = False

            

        
    if cell_start == True:
        time_cell_finish = datetime.now()
        time_cell_delta = (time_cell_finish - time_cell_start).total_seconds()
        #cv2.imwrite(events_folder_cell + '%s.jpg' % time_cell_finish, frame)
        
    else:
        if time_cell_delta >= min_tempoCELL:
            data = [[mac, datetime.now(),time_cell_delta, events_folder_cell,1]] ################# Error de tipeo
            #alerts.sendEmail(time_cell_delta, events_folder_cell, 1)
            #bank.sendData(data)
        elif time_cell_delta < min_tempoCELL and time_cell_delta != 0.0:
            if os.path.isdir(events_folder_cell) == True:
                shutil.rmtree(events_folder_cell)
            time_cell_delta = 0.0
    #print(cell_start)
    

