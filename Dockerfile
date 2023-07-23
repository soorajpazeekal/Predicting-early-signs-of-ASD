# Use the official Python image as the base image with Python 3.8
FROM python:3.8

# Set the working directory inside the container
WORKDIR /app

# Copy the contents of the "Predicting-early-signs-of-ASD" directory to the container's /app directory
COPY . /app

# Install necessary dependencies
RUN pip install -r requirements.txt

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN mkdir -p ~/.streamlit/
RUN echo "[general]"  > ~/.streamlit/credentials.toml
RUN echo "email = \"\""  >> ~/.streamlit/credentials.toml

# Expose the Streamlit port
EXPOSE 8501

# Set environment variable to ensure Streamlit doesn't run in headless mode
ENV STREAMLIT_SERVER_HEADLESS 0

# Set environment variable to configure Streamlit to run in the browser
ENV STREAMLIT_SERVER_ENABLE_CORS false

# Run your Streamlit app when the container starts
CMD ["streamlit", "run", "app.py", "--server.port", "8501"]
