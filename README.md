# Claysense

Claysense is a research project developed at Delft University of Technology as part of a graduation project. It focuses on real-time visual and machine learning-based 3D clay printing monitoring and parameter optimization to improve print quality and efficiency.

This repository contains the source code, data, and documentation for the Claysense system.


## Overview

Claysense integrates robotic control, computer vision, and deep learning techniques to automate the monitoring and adjustment of 3D clay printing processes. It employs DINOv2-based image feature extraction with a multi-head classifier predicting extrusion quality and overhang success parameters to enable closed-loop control.


## Folders

1. Model Training folder: Training related code and data
2. Training Result folder: Training output validation and visualization
3. UR5 folder: Robot control scripts and programs
4. Real-timecheck_workflow.py: real-time detection workflow

## Sample Test

You can test the samples in prediction_sample folder with the provided script and should receive an output similar to the one below.

```sh
python "Training Result/Samples_heatmap.py"
```

Example output:

```
********* Claysense sample predictions *********
Extrusion | Overhang
*********************************************

![Heatmap showing sample prediction results for extrusion and overhang quality in 3D clay printing. The heatmap displays color-coded regions indicating varying levels of print quality and success, with labeled axes for extrusion and overhang. The surrounding environment is a digital interface with a neutral, analytical tone. All text in the image is transcribed in the example output above.]("E:\OneDrive - Delft University of Technology\TUD Master\graduation project\report\Figure_1.png")
```

## Contact

For questions or feedback, contact Xiaochen Ding.