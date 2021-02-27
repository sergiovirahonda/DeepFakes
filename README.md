# DeepFakes
This repository is part of a project which seeks to build DIY deep fakes' models from scratch. Also it's part of an article that explains how to achieve such task. In this repo you'll contain the files required to preprocess videos for deep fakes generation, models training and swap faces in the final frames. Also, it shows you how to rebuild the original video with the transformed frames to finally get the deep fake. Finally, I show you how to use DeepFaceLab on Google Colab as an alternative to your DIY solution.

The repository contains several folders that contain the necessary files to achieve every stage of the project:

* **Deep faces dataset** contains the raw videos that I'll be using along this project but also has the workspace folder ready to be fed into DeepFaceLab.
* **Deep fakes - Docker files** contains the required files to build a Docker container to train your DIY solution on Google AI Platorm
* **Notebooks** contains all the Notebooks in .ipynb format used in this project: within DFL folder you'll find the resulting notebook that I used to obtain the Deep fake video with DeepFaceLab and within Kaggle folder you'll find all notebooks used for my DIY solution: data preprocessing, models training and face swapping / building the deep fake video.
* **Deep fakes - Final videos** contains the deep fake videos obtain with the DIY solution as well as with DeepFaceLab.

Find the fully interactive notebooks [here](https://www.kaggle.com/sergiovirahonda/deepfakes-preprocessing), [here](https://www.kaggle.com/sergiovirahonda/deepfakes-model-training), [here](https://www.kaggle.com/sergiovirahonda/deepfakes-face-swapping) and [here](https://colab.research.google.com/drive/1jSK1pxyc83SwkeIpGuGDk7p43T7P5NZM?usp=sharing).

Find the final models [here](https://www.kaggleusercontent.com/kf/54602307/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..hfyoww6gqfmRVBq-B3Kmnw.6f5odYX8_zZhQlLcylFj5va4rif00UqDEdWAXGiUypxooD8S9fIDveojg6JmPmxxyxMGe79doWrc7nbK76bAO8-HAsnaXLuxg64azpO_wAi-W1t7nNhch0HcrxxaOUYLXOTP3ewgoE6UJsp4rGtAAj90iLlP_tq4WnTd_zGhhPr4XRnBZF_g0FQG3IEOMBd99g6C9W-XRcgQ3Do5OHwGsgiNpT1mNe6Rnm6lecaB1FjD-KXe0gfeaBFWdb6Z2QeF6mpAPAvUUoeWS2T3IAbB4gy_BuMZ6m3dwgA3mtfRGX7qDGgPw_A9jEV0xGz-PLEbC_JV3C7A3SCxU6H5OroZBPYiJHk2lL24HDa4t2XY0NjMWXTTXWdCzbZidWGREN96jtUVEHj2EngbrJ_N8z3oazntANxdMuWGubtCZkUU9xkgmq9_EK5ULv7b0iq6KlBfggvXkA2oMZs38r0iLO7fahOJgHiyQknQpAG5TD6mib0h26OitORdEB-TmVfA6nrUypDJ93vLBlqIz6W5qUtoTLZIqZxozpStZVpLc54lnnv-xQbUSfGJVT-h2Rt_lJG9GMmc61gJ9LdzLSvTnyk-tqN4s4ddVq_p7r-AVwhU9FjsElQPbPuhVRWHTw_Ff7oNnLTdJJvwNTSGU04_dNVSLeam4p0Eb4CSgiUNxMb7RJ0._aMr_y0Vgfky3sBsU_zDlg/autoencoder_a.hdf5) and [here](https://www.kaggleusercontent.com/kf/54602307/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..hfyoww6gqfmRVBq-B3Kmnw.6f5odYX8_zZhQlLcylFj5va4rif00UqDEdWAXGiUypxooD8S9fIDveojg6JmPmxxyxMGe79doWrc7nbK76bAO8-HAsnaXLuxg64azpO_wAi-W1t7nNhch0HcrxxaOUYLXOTP3ewgoE6UJsp4rGtAAj90iLlP_tq4WnTd_zGhhPr4XRnBZF_g0FQG3IEOMBd99g6C9W-XRcgQ3Do5OHwGsgiNpT1mNe6Rnm6lecaB1FjD-KXe0gfeaBFWdb6Z2QeF6mpAPAvUUoeWS2T3IAbB4gy_BuMZ6m3dwgA3mtfRGX7qDGgPw_A9jEV0xGz-PLEbC_JV3C7A3SCxU6H5OroZBPYiJHk2lL24HDa4t2XY0NjMWXTTXWdCzbZidWGREN96jtUVEHj2EngbrJ_N8z3oazntANxdMuWGubtCZkUU9xkgmq9_EK5ULv7b0iq6KlBfggvXkA2oMZs38r0iLO7fahOJgHiyQknQpAG5TD6mib0h26OitORdEB-TmVfA6nrUypDJ93vLBlqIz6W5qUtoTLZIqZxozpStZVpLc54lnnv-xQbUSfGJVT-h2Rt_lJG9GMmc61gJ9LdzLSvTnyk-tqN4s4ddVq_p7r-AVwhU9FjsElQPbPuhVRWHTw_Ff7oNnLTdJJvwNTSGU04_dNVSLeam4p0Eb4CSgiUNxMb7RJ0._aMr_y0Vgfky3sBsU_zDlg/autoencoder_b.hdf5)

Find the transformed images and final video frames [here](https://drive.google.com/file/d/1_0JVo7bJ1l-y8Ar-oI1fsXuXtLxaKUsO/view?usp=sharing)

I hope this was helpful for you!
