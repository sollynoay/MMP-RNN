# Preparing ground truth MMP
## Optical flow estimation
Using [RAFT](https://github.com/princeton-vl/RAFT) for optical flow estimation. Pretrained weight for RAFT is [here](https://drive.google.com/file/d/1Q9_W23OaXrZO_AwXgLvNtpTFSFeROCBO/view?usp=sharing).
## Generating ground truth data
Run the code in MMP-generate.ipynb
## Our generated dataset
The dataset generated from GoPro dataset is also uploaded.  
[GTMMP](https://drive.google.com/file/d/1Z9xr6MFvuo9TMlT1wJMTscYBXuvEShO9/view?usp=sharing)  
It is worth mentioning that to avoid error caused by digitalization, MMPs were saved in .npy files instead of .png or .jpg. The size of the dataset is around 100G. One may also try generating MMP dataset from other high frequecy sharp datasets.  
