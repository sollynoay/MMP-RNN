# MMP-RNN 
## Datasets
[GOPRO](https://drive.google.com/file/d/1rDnbQV_YJtnAAXSG44lUWD_QzxQ9ppgL/view?usp=sharing)|[BSD](https://drive.google.com/file/d/1L6xHO9EPTk6LMEw_zs2suGWY56kNJas4/view?usp=sharing)
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
Results on GOPRO [here](https://drive.google.com/file/d/19LHci0U0xFiLuWjJ5w93mdB8zoHI8Grb/view?usp=sharing)  
Pretrained weight on GOPRO [here](https://drive.google.com/file/d/1xDosEjGCyDXRBKPMxJ1Vyp7nB1e2cygJ/view?usp=sharing)
## Reference
Code developed from [ESTRNN](https://github.com/zzh-tech/ESTRNN)

