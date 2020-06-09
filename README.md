# 3D Point Cloud and Cubic Volume generator
 Generate 3D point clouds (ply format) and 3D cubic volume files (.nii.gz format) for training using 3D U-Net.
## Background
 The Systems, Robotics and Vision (SRV) research group of Universitat de les Illes Balears (UIB) is taking part in a research project, called the TWIN roBOTs for cooperative underwater intervention missions (TWINBOT) Project, which has the purpose of improving Autonomous Underwater Vehicles (AUVs) used in survey missions. Further information can be found here: http://www.irs.uji.es/twinbot/twinbot.html#

Tasked with solving the object detection problems the robots will face, a project to implement an automatic 3D segmentation of underwater objects (for example, underwater pipes), i.e. identifying and labelling objects in a given surrounding, using Deep Learning (DL), is created.
This project comprises of:
- ROS for stereo image obtention and calibration.
- Python for disparity image, 3D point cloud, and volumetric data generation.
- 3D U-Net for training = automatic image segmentation/annotation

## Current project tasks
- Obtain disparity images and point clouds from calibrated stereo images.
Involves identifying identical features in both left and right stereo images, estimating the depth of the objects in the images and reproducing the 3D structures.
- Obtain ground truth images from disparity images.
With the help of an image editor, the pipes are identified and classified. This is, basically, a manual segmentation of the images.
- Obtain cubic volumes.
We calculate the missing depth values of the 2D stereo images to create a 3D cubic volume for training.

## Code overview 
### Disparity image - SGBM Tuning
A GUI that shows the SGBM parameters and computed disparity map in real time.

![alt text](https://github.com/pio-codes/3D-Point-Cloud-and-Cubic-Volume-generator/blob/master/gui.PNG?raw=true)

*Note: Ground truth image must be obtained from the generated disparity image*

### Point cloud generation
More information, in comparison to a standard .ply file, is added. The following figure shows the additional properties.

![alt text](https://github.com/pio-codes/3D-Point-Cloud-and-Cubic-Volume-generator/blob/master/ply.png?raw=true)

- X, Y, Z coordinates = 3D coordinates
- R, G, B original colour = RGB values from original image
- R, G, B ground truth colour = RGB values from objects in ground truth image
- Grayscale value = Grayscale value of objects
- X, Y disparity coordinates = For each 3D coordinate, its X, Y from original disparity map.
- Class ID = IDs/colour labels assigned to the objects during ground truth generation. Based on the following assumed classes:

![alt text](https://github.com/pio-codes/3D-Point-Cloud-and-Cubic-Volume-generator/blob/master/colourmap.png?raw=true)

The generated point cloud can be viewed with the following Python script: `view_point_cloud`

## Requirements
- INPUT DATA
  - Calibrated stereo images (left and right image)
  - Ground truth images of each pair
 
 ## Citation/Credits
 Some of the codes include highly modified version of the [***StereoVision***](https://github.com/erget/StereoVision) package by Daniel Lee.
