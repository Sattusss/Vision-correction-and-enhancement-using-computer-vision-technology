#!/usr/bin/env python
import PySimpleGUI as sg
import cv2 as cv
import tensorflow_hub as hub
import tensorflow as tf
import pandas as pd
import pyttsx3
import threading
import time



def voice_alarm(alarm_sound: pyttsx3.Engine,text):
    alarm_sound.say(f'{text} detected')
    alarm_sound.runAndWait()
    time.sleep(5)

def main():
    
    alarm_sound = pyttsx3.init()
    voices = alarm_sound.getProperty('voices')
    alarm_sound.setProperty('voice', voices[0].id)
    alarm_sound.setProperty('rate', 150)

    detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/lite2/detection/1")
    labels = pd.read_csv('labels.csv',sep=';',index_col='ID')
    labels = labels['OBJECT (2017 REL.)']

    vidFile = cv.VideoCapture(0)
    fps = vidFile.get(cv.CAP_PROP_FPS)
    sg.theme('Black')
    # ---===--- define the window layout --- #
    layout = [
        [sg.Text('Project', size=(15, 1), font='Helvetica 20')],
        [sg.Image(key='-IMAGE-')],
        [sg.Push(), sg.Button('Exit', font='Helvetica 14')]
    ]

    window = sg.Window('Application', layout, no_titlebar=False, location=(0, 0))
    image_elem = window['-IMAGE-']
    timeout = 1000//fps                 # time in ms to use for window reads
    cur_frame = 0
    width = 640
    height = 420
    while vidFile.isOpened():
        event, values = window.read(timeout=timeout)
        if event in ('Exit', None):
            break
        ret, frame = vidFile.read()
        if not ret:  # if out of data stop looping
            break
        # if someone moved the slider manually, the jump to that frame
        inp = cv.resize(frame, (width , height ))

        #Convert img to RGB
        rgb = cv.cvtColor(inp, cv.COLOR_BGR2RGB)

        #Is optional but i recommend (float convertion and convert img to tensor image)
        rgb_tensor = tf.convert_to_tensor(rgb, dtype=tf.uint8)

        #Add dims to rgb_tensor
        rgb_tensor = tf.expand_dims(rgb_tensor , 0)
        
        boxes, scores, classes, num_detections = detector(rgb_tensor)
        
        pred_labels = classes.numpy().astype('int')[0]
        
        pred_labels = [labels[i] for i in pred_labels]
        pred_boxes = boxes.numpy()[0].astype('int')
        pred_scores = scores.numpy()[0]
        for score, (ymin,xmin,ymax,xmax), label in zip(pred_scores, pred_boxes, pred_labels):
            if score < 0.5:
                continue  
            score_txt = f'{100 * round(score,0)}'
            rgb = cv.rectangle(rgb,(xmin, ymax),(xmax, ymin),(0,255,0),3)      
            font = cv.FONT_HERSHEY_SIMPLEX
            cv.putText(rgb,label,(xmin, ymax-10), font, 0.5, (255,0,0), 1, cv.LINE_AA)
            cv.putText(rgb,score_txt,(xmax, ymax-10), font, 0.5, (255,0,0), 1, cv.LINE_AA)
            if label != 'person':
                alarm = threading.Thread(target=voice_alarm, args=(alarm_sound,label))
                alarm.start()
        rgb = cv.cvtColor(rgb, cv.COLOR_RGB2BGR)
        cur_frame += 1
        imgbytes = cv.imencode('.ppm', rgb)[1].tobytes()  # can also use png.  ppm found to be more efficient
        image_elem.update(data=imgbytes)

main()