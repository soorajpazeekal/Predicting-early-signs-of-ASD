# Predicting-early-signs-of-ASD
an applied research project with a full-featured web application to screen and predict early possible signs of ASD (autism spectrum disorder).
## What is ASD exactly means?
Autism is a severe developmental spectrum disorder that puts constraints on communicating linguistic, cognitive, and social interaction skills. Autism spectrum disorder screening detects potential autistic traits in an individual where the early diagnosis shortens the process and has more accurate results. The methods used to predict Autism by doctors involve physical identification of facial features, questioners, Fine motor skills, MRI scans, etc. This conventional diagnosis method needs more time, cost and in the case of pervasive developmental disorders, the parents feel inferior to come out in the open. Therefore, it is close to using a timely ASD test that helps assist health professionals and informs people whether they should follow a formal clinical diagnosis or not. A diagnostic tool that can identify the risk of ASD during childhood provides an opportunity for intervention before full symptoms. The proposed model uses a convolution neural network classifier that helps predict the early autistic traits in children through facial features in images, with the least cost, lesstime, and a more significant accuracy than the traditional type of diagnosis.

## Signs and Symptoms of Autism Spectrum Disorder
### Social Communication and Interaction Skills
Social communication and interaction skills can be challenging for people with ASD.
### Restricted or Repetitive Behaviors or Interests etc.
People with ASD have behaviors or interests that can seem unusual. These behaviors or interests set ASD apart from conditions defined by problems with social communication and interaction only.

![Screenshot](/screenshots/image_86ad5d3b8a.png)

# Simple data pipeline (From capureing video then to extracting video frames.)
![Screenshot](/screenshots/Screenshot%20(208).png)

# sample video for data pre-processing and testing. for more checkout 'videos' folder!
![video](https://github.com/soorajpazeekal/Predicting-early-signs-of-ASD/blob/development/videos/VIDEO_Autistic_63abfea1-e5f9-4230-9f8d-9ce851579555.mp4)

# Installation
This web application requires streamlit, python >= 3.8+, keras and tensorflow backend. conda enviroment is recommend rather than any other env programes.

## Install with docker
 - Please make sure docker installed on your system
 - Run these commands with in the root folder of this project
     ```sh
    docker build -t my_streamlit_app .
    ```
     ```sh
    docker run --name test_container -p 8501:8501 my_streamlit_app
    ```
    ```sh
    docker start test_container
    ```

## Build from sourcecode
 - Install all dependencies with conda (make sure Anaconda or Miniconda is installed)
 
     ```sh
    git clone https://github.com/soorajpazeekal/Predicting-early-signs-of-ASD.git
    ```
    ```python
    conda env create --name envname --file=environments.yml
    ```
    ```python
    conda activate {envname}
    ```
 - Make sure tensorflow version == 2.8.2
    ```python
    streamlit run app.py
    ```
# Screenshots
![Screenshot](/screenshots/Opera%20Snapshot_2022-08-29_214151_sharp-mayfly-96.loca.lt.png)
![Screenshot](/screenshots/Screenshot%20(211).png)
![Screenshot](/screenshots/Screenshot%20(204).png)
