# Efficient Video Deblurring Guided by Motion Magnitude
Official repository of MMP-RNN (ECCV 2022)
<div align="center"><img src="https://user-images.githubusercontent.com/11170161/178935637-6bb6a25a-dc67-4d5e-9c3c-f7f31be41085.png" width="640"></div>


# Preparing ground truth motion magnitude prior
<div align="center"><img src="https://user-images.githubusercontent.com/11170161/178949402-2de1df49-4fd8-481c-a4c2-8a0c534fa0fe.png" width="640"></div>
The ground truth MMP is generated from high-frequency sharp frames during exposure time. Optical flows between sharp frames are estimated by [RAFT](https://github.com/princeton-vl/RAFT). 
