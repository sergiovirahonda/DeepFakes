# DeepFakes
This repository is part of a project which seeks to build DIY deep fakes' models from scratch. Also it's part of an article that explains how to achieve such task. In this repo you'll contain the files required to preprocess videos for deep fakes generation, models training and swap faces in the final frames. Also, it shows you how to rebuild the original video with the transformed frames to finally get the deep fake. Finally, I show you how to use DeepFaceLab on Google Colab as an alternative to your DIY solution.

The repository contains several folders that contain the necessary files to achieve every stage of the project:

* **Deep faces dataset** contains the raw videos that I'll be using along this project but also has the workspace folder ready to be fed into DeepFaceLab.
* **Deep fakes - Docker files** contains the required files to build a Docker container to train your DIY solution on Google AI Platorm
* **Notebooks** contains all the Notebooks in .ipynb format used in this project: within DFL folder you'll find the resulting notebook that I used to obtain the Deep fake video with DeepFaceLab and within Kaggle folder you'll find all notebooks used for my DIY solution: data preprocessing, models training and face swapping / building the deep fake video.
* **Deep fakes - Face swapping and video rebuilding** contains the final output of our last notebook. There you'll find the final frames, the transformed faces and so on.
* **Deep fakes - Final videos** contains the deep fake videos obtain with the DIY solution as well as with DeepFaceLab.

Find the fully interactive notebooks [here](https://www.kaggle.com/sergiovirahonda/deepfakes-preprocessing), [here](https://www.kaggle.com/sergiovirahonda/deepfakes-model-training), [here](https://www.kaggle.com/sergiovirahonda/deepfakes-face-swapping) and [here](https://colab.research.google.com/drive/1jSK1pxyc83SwkeIpGuGDk7p43T7P5NZM?usp=sharing).

I hope this was helpful for you!