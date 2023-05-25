# MMP-RNN 
## Datasets
[GOPRO](http://gofile.me/7aSbh/XdCPL5WWD)|[BSD](http://gofile.me/7aSbh/svN9MpyHS)
## Train
```bash
python main.py
```
## Test only
In main.py
```bash
para.test_only = True
para.test_save_dir = './results/'
para.test_checkpoint = './experiment/<weight>'
```
## Weight for MMP-Net
The uploaded weight is for MMP-Net, a modified UNet for MMP estimation.  
To download the pretrained weight for MMP-RNN, please check the link follows.
## Results
Deblurring results on GOPRO [here](http://gofile.me/7aSbh/musaTkZFo)  
Pretrained weight on GOPRO [here](https://drive.google.com/file/d/1xDosEjGCyDXRBKPMxJ1Vyp7nB1e2cygJ/view?usp=sharing)  
Deblurring results on BSD [here](http://gofile.me/7aSbh/Qt45dOODN)  
Pretrained weight on BSD [here](https://drive.google.com/file/d/11eml6TTVS_bfBzVy5QGa84VlvDnGlhUY/view?usp=share_link) 
## Reference
Code developed from [ESTRNN](https://github.com/zzh-tech/ESTRNN)

