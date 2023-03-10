import cv2
import numpy as np
import json

f = open('parameters.json')
parameters = json.load(f)
scale = parameters['escala']

gancho_altura_k = parameters['gancho_altura_k']
gancho_altura_alpha = parameters['gancho_altura_alpha']
gancho_elipse_k = parameters['gancho_elipse_k']
gancho_elipse_razao = parameters['gancho_elipse_razao']
gancho_elipse_alpha_x = parameters['gancho_elipse_alpha_x']
gancho_elipse_alpha_y = parameters['gancho_elipse_alpha_y']
tolerancia_bb_pessoa_a = parameters['tolerancia_bb_pessoa_a']
tolerancia_bb_pessoa_b = parameters['tolerancia_bb_pessoa_b']

def checkPoint(xc, yc, x, y, a, b):
    #x²/a² + y²/b² <= 1
    z = ((x-xc)/a)**2 + ((y-yc)/b)**2
    if z <= 1:
        inside = True
    else:
        inside = False

    return inside

N_step = 100
n_step  = 0
def riskZone(image, bb_list):
    global n_step, N_step

    zone_color = (0,255,255)

    people = []
    gancho = []
    if len(bb_list) > 0:
        for bb in bb_list:
            x1 = int(bb['xmin']*scale)
            y1 = int(bb['ymin']*scale)
            x2 = int(bb['xmax']*scale)
            y2 = int(bb['ymax']*scale)
            label = int(bb['label'])
            if label == 5:
                people.append([x1,y1,x2,y2])
            if label == 2:
                gancho = [x1,y1,x2,y2]

    if len(gancho) > 0:
        xc_gancho = int(0.5*(gancho[0]+gancho[2]))
        yc_gancho = int(0.5*(gancho[1]+gancho[3]))
        h_gancho = int(gancho_altura_k*(xc_gancho**gancho_altura_alpha))
        a_gancho = int(gancho_elipse_k*(xc_gancho**gancho_elipse_alpha_x*10000)/(1 + yc_gancho**gancho_elipse_alpha_y))
        b_gancho = int(gancho_elipse_razao*gancho_elipse_k*(xc_gancho**gancho_elipse_alpha_x)*10000/(1 + yc_gancho**gancho_elipse_alpha_y))
        p1 = (xc_gancho,yc_gancho)
        p2 = (xc_gancho,h_gancho)

        distances = []
        for person in people:
            xc = 0.5*(person[0] + person[2])
            yc = 0.5*(person[1] + person[2])
            distance = (xc - xc_gancho)**2 + (yc - h_gancho)**2
            distances.append(distance)

        person_in_risk = False
        for person in people:
            xc = x1 = person[0]
            y1 = person[1]
            x2 = person[2]
            y2 = person[3]
            xc = int(0.5*(x1+x2))

            risco1 = checkPoint(xc_gancho, h_gancho, xc,y2-tolerancia_bb_pessoa_a, a_gancho, b_gancho)
            risco2 = checkPoint(xc_gancho, h_gancho, xc,y2-tolerancia_bb_pessoa_b, a_gancho, b_gancho)

            if risco1 == True or risco2 == True:
                zone_color = (0,0,255)
                person_in_risk = True

        overlay = image.copy()
        cv2.circle(overlay, p1, 20, (0,0,255), -1)
        cv2.circle(overlay, p2, 20, (255,0,0), -1)
        square_cnt = np.array( [(xc_gancho-a_gancho,yc_gancho), (xc_gancho+a_gancho,yc_gancho), (xc_gancho+a_gancho, h_gancho), (xc_gancho-a_gancho, h_gancho)] )
        cv2.drawContours(overlay, [square_cnt], 0, zone_color, -1)
        cv2.ellipse(overlay, p1, (a_gancho, b_gancho), 0, 0, 360, zone_color, -1)
        cv2.ellipse(overlay, p1, (a_gancho, b_gancho), 0, 0, 360, (0,100,100), 3)

        cv2.ellipse(overlay, p2, (a_gancho, b_gancho), 0, 0, 360, zone_color, -1)
        cv2.ellipse(overlay, p2, (a_gancho, b_gancho), 0, 0, 360, (0,100,100), 3)
        cv2.line(overlay, p1, p2, (0,0,255), 10)

        alpha = 0.25
        overlay = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
        image = 1*overlay

    else:
        person_in_risk = False
        distances = []

    for pessoa in people:
        xmin = pessoa[0]
        ymin = pessoa[1]
        xmax = pessoa[2]
        ymax = pessoa[3]
        cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (0,0,255), 2)

    return image, person_in_risk, distances
