FROM gcr.io/deeplearning-platform-release/tf2-gpu.2-3
WORKDIR /root

RUN pip install pandas numpy google-cloud-storage scikit-learn opencv-python

RUN apt-get update; apt-get install git -y; apt-get install -y libgl1-mesa-dev
RUN git clone https://github.com/sergiovirahonda/DeepFakesDataset.git

COPY model.py ./model.py
COPY data_utils.py ./data_utils.py
COPY task.py ./task.py

ENTRYPOINT ["python","task.py"]
