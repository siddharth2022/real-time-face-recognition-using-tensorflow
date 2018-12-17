from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import time

import cv2
import numpy as np
import tensorflow as tf
from scipy import misc
import os

import math
from matplotlib import pyplot as plt

from packages import facenet, detect_face
from packages.preprocess import preprocesses
import sys

from packages.classifier import training
cur_dir = os.getcwd()
print(cur_dir)
mode = 'd'
modeldir = './model'
classifier_filename = './class/classifier.pkl'
npy=''
train_img="./train_img"


while True:
    os.chdir(cur_dir)
    mode = input("enter:")
    if mode == 't':
        input_datadir = './train_img'
        output_datadir = './pre_img'

        obj=preprocesses(input_datadir,output_datadir)
        nrof_images_total,nrof_successfully_aligned=obj.collect_data()

        print('Total number of images: %d' % nrof_images_total)
        print('Number of successfully aligned images: %d' % nrof_successfully_aligned)
        datadir = './pre_img'
        modeldir = './model'
        classifier_filename = './class/classifier.pkl'
        print ("Training Start")
        obj=training(datadir,modeldir,classifier_filename)
        get_file=obj.main_train()
        print('Saved classifier model to file "%s"' % get_file)
        

    elif mode == 'n':
        name = input("Enter Your Name:")
        face_cascade = cv2.CascadeClassifier('lib\haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier('lib\haarcascade_eye.xml')

        cap = cv2.VideoCapture(0)

        scale_factor = 1.2
        min_neighbours = 5
        min_size = (30,30)
        biggest_only = True
        ff = cv2.CASCADE_FIND_BIGGEST_OBJECT | \
                cv2.CASCADE_DO_ROUGH_SEARCH if biggest_only else \
                cv2.CASCADE_SCALE_IMAGE
        counter = 0
        os.chdir('./train_img')
        if os.path.exists(name):
            os.chdir(name)
        else:
            os.mkdir(name)
            os.chdir(name)
                    
        while(True):
            
            ret, img = cap.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            faces = face_cascade.detectMultiScale(gray, 1.3, 5,minSize=min_size,flags=ff)
            for (x,y,w,h) in faces:
                    
                    y1=int(y-h*.1)
                    y2=int(y+h*1.1)
                    x1=int(x-w*.1)
                    x2=int(x+w*1.1)
                    img1=gray[y1:y2 , x1:x2]
                    img1 = cv2.equalizeHist(img1)
                    counter=counter+1
                    while True:
                        if os.path.exists(str(counter)+'.jpg'):
                            counter=counter+1
                        else:
                            break
                    cv2.imwrite(str(counter)+'.jpg',img)
                    cv2.imshow('frame3',img1)
                    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                    roi_gray = gray[y:y+h, x:x+w]
                    roi_color = img[y:y+h, x:x+w]
                    eyes = eye_cascade.detectMultiScale(roi_gray)
                    for (ex,ey,ew,eh) in eyes:
                        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                    
                    
            cv2.imshow('frame1',img)
            

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        
        cap.release()
        cv2.destroyAllWindows()
        
    
    elif mode == 'd' :
            with tf.Graph().as_default():
                gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
                sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
                with sess.as_default():
                    pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)

                    minsize = 20  # minimum size of face
                    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
                    factor = 0.709  # scale factor
                    margin = 44
                    frame_interval = 3
                    batch_size = 1000
                    image_size = 182
                    input_image_size = 160
                    
                    HumanNames = os.listdir(train_img)
                    HumanNames.sort()

                    print('Loading Modal')
                    facenet.load_model(modeldir)
                    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                    embedding_size = embeddings.get_shape()[1]


                    classifier_filename_exp = os.path.expanduser(classifier_filename)
                    with open(classifier_filename_exp, 'rb') as infile:
                        (model, class_names) = pickle.load(infile)

                    video_capture = cv2.VideoCapture(0)
                    c = 0
                    #cv2.namedWindow('Video',cv2.WINDOW_NORMAL)
                    #cv2.resizeWindow('Video', 900,500)
                    
                    print('Start Recognition')
                    prevTime = 0
                    while True:
                        ret, frame = video_capture.read()

                        #frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)    #resize frame (optional)

                        curTime = time.time()+1 
                        timeF = frame_interval

                        if (c % timeF == 0):
                            find_results = []

                            if frame.ndim == 2:
                                frame = facenet.to_rgb(frame)
                            frame = frame[:, :, 0:3]
                            bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                            nrof_faces = bounding_boxes.shape[0]
                            print('Detected_FaceNum: %d' % nrof_faces)

                            if nrof_faces > 0:
                                det = bounding_boxes[:, 0:4]
                                img_size = np.asarray(frame.shape)[0:2]

                                cropped = []
                                scaled = []
                                scaled_reshape = []
                                bb = np.zeros((nrof_faces,4), dtype=np.int32)

                                for i in range(nrof_faces):
                                    emb_array = np.zeros((1, embedding_size))

                                    bb[i][0] = det[i][0]
                                    bb[i][1] = det[i][1]
                                    bb[i][2] = det[i][2]
                                    bb[i][3] = det[i][3]

                                    # inner exception
                                    if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                                        print('Face is very close!')
                                        continue

                                    cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                                    cropped[i] = facenet.flip(cropped[i], False)
                                    scaled.append(misc.imresize(cropped[i], (image_size, image_size), interp='bilinear'))
                                    scaled[i] = cv2.resize(scaled[i], (input_image_size,input_image_size),
                                                           interpolation=cv2.INTER_CUBIC)
                                    scaled[i] = facenet.prewhiten(scaled[i])
                                    scaled_reshape.append(scaled[i].reshape(-1,input_image_size,input_image_size,3))
                                    feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                                    emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                                    predictions = model.predict_proba(emb_array)
                                    print(predictions)
                                    best_class_indices = np.argmax(predictions, axis=1)
                                    best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                                    # print("predictions")
                                    print(best_class_indices,' with accuracy ',best_class_probabilities)

                                    # print(best_class_probabilities)
                                    if best_class_probabilities>0.53:
                                        cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)    #boxing face

                                        #plot result idx under box
                                        text_x = bb[i][0]
                                        text_y = bb[i][3] + 20
                                        print('Result Indices: ', best_class_indices[0])
                                        print(HumanNames)
                                        for H_i in HumanNames:
                                            if HumanNames[best_class_indices[0]] == H_i:
                                                result_names = HumanNames[best_class_indices[0]]
                                                cv2.putText(frame, result_names, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                            1, (0, 0, 255), thickness=1, lineType=2)
                            else:
                                print('Alignment Failure')
                        # c+=1
                        cv2.imshow('Video', frame)

                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                        
                    video_capture.release()
                    cv2.destroyAllWindows()




