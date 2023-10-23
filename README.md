# AD_reference_code
### Overview
This code is our implementation fo the the PatchCore model for anomaly detection on the MVTec dataset that was discussed in the paper.

### Results
The results that were discussed in our paper can be found in the results folder. 
It contain 4 differnt folders:
- AAResNet: Results to antialiasing resnet PatchCore
- No augmentation: Results to the orignal PatchCore 
- Pseudo Label: Results to the pseudo label PatchCore
- WideResNet: Results to PatchCore with data augmented 

### Dataset
The code will automatically download MVTecAD. 
However, please do [downlaod](https://drive.google.com/file/d/1x0RDvMPooWdE8WcVbEZ5OveyvCWUKNcU/view?usp=sharing) the dataset that we refered to in the paper. 

## Running the Script
To run the script, use the following command in the working dir of the code:
```bash
python run.py --dataset <dataset> --backbone <backbone>
```
By default the dataset will be all classes in mvtec and the wide_resnet50_2 backbone.

replace <dataset> wiht any of these to run specific datasets that were mention in our paper.
- all: all MVTec Classes
- exp_0_nut: orignal metal nut one shot dataset (No aug)
- exp_0_screw: orignal screw nut one shot dataset (No aug)
- exp_1_nut: metal nut dataset with 1:3 data aug
- exp_1_screw: screw dataset with 1:3 data aug
- exp_2_nut: metal nut dataset with all data aug
-  exp_2_screw: screw dataset with all data aug
- exp_3_nut: metal nut dataset with pseudo labels
- exp_3_screw: screw dataset with pseudo labels
- 
### References
The original open source code that we adapt from was based off:
https://github.com/rvorias/ind_knn_ad
