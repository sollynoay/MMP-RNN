# Preparing ground truth MMP
## Optical flow estimation
Using [RAFT](https://github.com/princeton-vl/RAFT) for optical flow estimation. Pretrained weight for RAFT is [here](https://drive.google.com/file/d/1Q9_W23OaXrZO_AwXgLvNtpTFSFeROCBO/view?usp=sharing).
## Generating ground truth data
Run the code in MMP-generate.ipynb
## Our generated dataset
The dataset generated from GoPro dataset is also uploaded [here](https://drive.google.com/file/d/1Z9xr6MFvuo9TMlT1wJMTscYBXuvEShO9/view?usp=sharing).  
[GoPro_Large_all](https://seungjunnah.github.io/Datasets/gopro.html) was used for dataset generation.  
It is worth mentioning that to avoid error caused by digitalization, MMPs were saved in .npy files instead of .png or .jpg. The size of the dataset is around 100G. One may also try generating MMP dataset from other high frequecy sharp datasets.  
## Tips for training  
Larger crop size may lead to better convergence during training. We tried (512,512) and witnessed good convergence. In addition, feeding all the data to the network may not lead to convergence. It is found that trimming the full dataset to 75% or 50% may lead to better convergence.  
