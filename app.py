import streamlit as st
import time
import cv2
import os
import random

video_directory = "videos"
video_frames_directory = "video_frames"
pretrained_model = "pretrained_models"


st.title("✔️ Predicting early signs of autism using facial landmarks with series of video frames")
with st.expander("📙 See more info"):
    st.write("""Autism is a severe developmental spectrum disorder that puts constraints on communicating linguistic, 
cognitive, and social interaction skills. Autism spectrum disorder screening detects potential 
autistic traits in an individual where the early diagnosis shortens the process and has more accurate results. 
The methods used to predict Autism by doctors involve physical identification of facial features, questioners, 
Fine motor skills, MRI scans, etc. This conventional diagnosis method needs more time, cost and in the case of 
pervasive developmental disorders, the parents feel inferior to come out in the open. Therefore, it is close to 
using a timely ASD test that helps assist health professionals and informs people whether they should follow a 
formal clinical diagnosis or not. A diagnostic tool that can identify the risk of ASD during childhood provides an o
pportunity for intervention before full symptoms. The proposed model uses a convolution neural network classifier 
that helps predict the early autistic traits in children through facial features in images, with the least cost, less 
time, and a more significant accuracy than the traditional type  of diagnosis.""")

st.subheader("📍 1)First, => Upload / select a video:")

#col1, col2, col3 = st.columns(3)


with st.sidebar:
  st.write("General informations")
  with st.expander("👍 List of avaliable demo videos: "):
      st.write("""Specifically, all copied material is owned by the respective authors. 
      These are publicly available videos (ie: on YouTube).""")
      for item in os.listdir(os.path.join(video_directory)):
          if os.path.exists('.ipynb_checkpoints'):
              import shutil
              shutil.rmtree(".ipynb_checkpoints")
          st.write(f"Filename: {item}")
          st.video(os.path.join(f"{video_directory}/")+item)

  with st.expander("🔥 View pretrained model performance: "):
      st.write("The accuracy of the model is = 87.5 %")
      st.write("train.class_indices : {'Autistic': 0, 'Non_Autistic': 1}")
      st.image(f"{pretrained_model}/model_performance.png")

  with st.expander("🔥 List of cropped faces from the video "):
    try:
      for item in os.listdir(os.path.join(video_frames_directory)): 
        st.write(f"Filename: {item}")
        st.image(os.path.join(f"{video_frames_directory}/")+item)
    except:
      st.write("""You can preview the already cropped faces from your input video here. 
      If it's empty, please run the first and second steps instead! """)  



option = st.selectbox(
     'Select from list of videos:',
     (os.listdir(os.path.join(video_directory))))

st.video(os.path.join(f"{video_directory}/")+option)


model_to_img = cv2.CascadeClassifier(f'{pretrained_model}/haarcascade_frontalface_default.xml')


vid = cv2.VideoCapture(os.path.join(f"{video_directory}/")+option)


uploaded_file = st.file_uploader("or Upload a video")
if uploaded_file is not None:
     # To read file as bytes:
     st.write("Filename: ", uploaded_file)
     with open(os.path.join(video_directory, uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())
        st.success("Saved File:{} to videos".format(uploaded_file.name))





# get total number of frames
totalFrames = vid.get(cv2.CAP_PROP_FRAME_COUNT)
st.subheader("📍 2)Second, => Extract facial landmarks from video and feed them to the model.")
with st.expander("See more info"):
  st.write("""Before running the model, please make sure all your faces are cropped perfectly! 
  You can preview the data before making any decisions.""")
selected_Frames = st.slider("""🔴 Optional: You can select how many total random frames you want to analyse. The default value is 15; 
larger frame numbers may take a long time to compute! It will depend on your machine's GPU or CPU. (If it is Offline)""", 
min_value=15, max_value=int(totalFrames), value=15, help=f"A total of {totalFrames} frames were counted. and the default value chosen is: 15")
if st.button('Start'):  
  st.write(f"Total number of frames in this video: {totalFrames}")
  if os.path.exists(video_frames_directory):
    import shutil
    shutil.rmtree(video_frames_directory)
  if not os.path.exists(video_frames_directory):
    os.makedirs('video_frames')
  for i in range(selected_Frames):
    randomFrameNumber=random.randint(0, totalFrames)
    # set frame position
    vid.set(cv2.CAP_PROP_POS_FRAMES,randomFrameNumber)
    success, image = vid.read()
    if success:
      #cv2.imwrite("random_frame_"+str(randomFrameNumber)+".jpg", image)
      face  = model_to_img.detectMultiScale(image)
      if len(face) == 0:
        print("no face")
      else:
        x1 = face[0][0]
        y1 = face[0][1]
        x2 = face[0][2] + x1
        y2 = face[0][3] + y1 
        crop_img = image[y1:y2 , x1:x2]         
        #cv2_imshow(crop_img)
        cv2.imwrite(f"{video_frames_directory}/random_frame_"+str(randomFrameNumber)+".jpg",crop_img)
  with st.expander("Tap to see extracted faces:"):
    st.write("Selected Faces: ")
    for item in os.listdir(os.path.join(video_frames_directory)): 
      st.image(os.path.join(f"{video_frames_directory}/")+item) 
  st.success("✅ successfully extracted face points from the input video. Saved to the selected folder!")  




import numpy as np
import pandas as pd
import os

import keras
import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions

from keras.models import load_model



st.subheader("📍 3)Third => Finally, run the model.")
with st.expander("See more info"):
  st.write("""This section can run the ML model and predict the valuable outcomes as a result. 
  First, run the model and examine the results!""")
if st.button('Run model'):
  with st.spinner('Wait for it...'):
      model = load_model(os.path.join(f"{pretrained_model}/best_model.h5"))
      print(len(os.listdir(video_frames_directory)))
      list_pred = []
      for item in os.listdir(video_frames_directory):
          print(item)
          img = load_img(f"{video_frames_directory}/"+item, target_size= (256,256))
          i = img_to_array(img)
          im = preprocess_input(i)
          img = np.expand_dims(im, axis= 0)
          pred = np.argmax(model.predict(img))
          print(f"model prediction is: {pred}")
          list_pred.append(pred)
      st.write("Predicted results: ")  
      list_pred  
      #st.write(f"Autistic: {list_pred.count(0)}, Non_Autistic: {list_pred.count(1)}") 
      st.write(max(set(list_pred), key=list_pred.count))
      col1, col2 = st.columns(2)
      with col1:
          st.header("Total Autistic signs were discovered?")
          st.metric(label="Autistic:", value=list_pred.count(0), delta=list_pred.count(1))
      with col2:
          st.header("Total Non_Autistic signs were discovered?")
          st.metric(label="Non_Autistic:", value=list_pred.count(1), delta=f"-{list_pred.count(0)}")  

