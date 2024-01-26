FROM python:3.7.16

RUN apt-get update && apt-get install -y gcc

RUN apt-get install -y ffmpeg 
RUN apt-get install -y libsm6 
RUN apt-get install -y libxext6  


RUN pip install setuptools cython numpy

WORKDIR /convert_app

COPY . .

# installing neccessary packages
RUN python -m pip install --upgrade pip
RUN pip install future-fstrings 
RUN pip install mathutils==2.81.2
RUN pip install bpy-2.91a0-cp37-cp37m-manylinux2014_x86_64.whl && bpy_post_install
RUN pip install streamlit==1.23.1
RUN pip install click==8.1.7
RUN pip install protobuf==4.24.4

EXPOSE 8501

# RUN export TKINTER_HEADLESS=1

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "tools/app_stml.py", "--server.port=8501", "--server.address=0.0.0.0"]

