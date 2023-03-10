import cv2
import numpy as np
from datetime import datetime, timedelta
import os
import shutil
import sys
from flask import Flask, request, jsonify, Response, render_template
from flask_restful import Resource, Api, reqparse
import json

import model
import alerts
import zone
import bank

f = open('parameters.json')
parameters = json.load(f)

app = Flask(__name__)
api = Api(app)
source = cv2.VideoCapture(parameters['camera1'])

data_evento = datetime.now()
mac = parameters['mac']
scale = parameters['escala']
min_tempoArea = parameters['min_tempoArea']
porta = parameters['porta']

startCount = False
risk_time = 0
imgs = []
frame_counter = 0
events_folder = './data/eventos/None' 
send_data = False
bb_list = []
risk_t0 = 0

frame = np.zeros((600,800,3)).astype(int)
frame_zero = np.zeros((600,800,3)).astype(int)

class getFrame(Resource):
    def get(self):
        global frame, risk_time, startCount, imgs, frame_counter, events_folder, send_data, bb_list, risk_t0
        
        while True:
            success, img = source.read()
            img = cv2.resize(img, (int(scale*img.shape[1]), int(scale*img.shape[0])))
            #frame = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            frame = img
            bb_list = model.predict(frame)
            _, risco, distances = zone.riskZone(frame, bb_list)

            model.checkEPI(frame, bb_list)
            
            '''if len(distances) == 0:
                startCount = False
                #alerts.blink_mode('wait')
                #alerts.soundAlert('wait')
        
            if len(distances) !=0 and risco == False:
                startCount = False
                #alerts.blink_mode('attention')
                #alerts.soundAlert('attention')
                
            if len(distances) !=0 and risco == True:
                if startCount == False:
                    risk_t0 = datetime.now()
                    events_folder = './data/eventos/%s' % risk_t0
                    os.mkdir(events_folder)
                startCount = True
                #alerts.blink_mode('alert')
                #alerts.soundAlert('alert')
            
            if startCount == True:
                risk_t1 = datetime.now()
                risk_time =  (risk_t1 - risk_t0).total_seconds()
                frame_counter += 1
                if frame_counter == 5:
                    img, _, _ = zone.riskZone(frame, bb_list)
                    cv2.imwrite(events_folder + '/%s.jpg' % risk_t1, img)
                    frame_counter = 0
                
            if startCount == False and risk_time >= min_tempoArea:
                send_data = True
            elif startCount == False and risk_time < min_tempoArea:
                if os.path.isdir(events_folder) == True:
                    shutil.rmtree(events_folder)
                    send_data = False
                
            if send_data == True:
                print('alerta', risk_time)
                data = [[mac,datetime.now(),risk_time,'%s' % events_folder,5]]
                bank.sendData(data)
                alerts.sendEmail(risk_time, events_folder, 5)
                send_data = False
                risk_time = 0.0
                events_folder = './data/eventos/None'   '''
  
def gen_frames(camera_id):
    global frame, bb_list
    camera_id = int(camera_id)
    
    if camera_id == 0:
        while True:
            img0, risco, distances = zone.riskZone(frame, bb_list)
            ret, buffer = cv2.imencode('.jpg', img0)
            img0 = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + img0 + b'\r\n')
            
    if camera_id == 1:
        while True:
            img1 = model.draw_icons(frame,bb_list)
            ret, buffer = cv2.imencode('.jpg', img1)
            img1 = buffer.tobytes()

            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + img1 + b'\r\n')

@app.route('/video_feed/<string:list_id>/', methods=["GET"])
def video_feed(list_id):
    return Response(gen_frames(list_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/', methods=["GET"])
def index():
    return render_template('index.html', camera_list=2)

api.add_resource(getFrame, '/getFrame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=porta, threaded=True, use_reloader=False)
