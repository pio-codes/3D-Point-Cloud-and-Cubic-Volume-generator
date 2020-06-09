# 3D Point Cloud and Cubic Volume generator
 Generate 3D point clouds (ply format) and 3D cubic volume files (.nii.gz format) for training using 3D U-Net.
# Background
 The Systems, Robotics and Vision (SRV) research group of Universitat de les Illes Balears (UIB) is taking part in a research project, called the TWIN roBOTs for cooperative underwater intervention missions (TWINBOT) Project, which has the purpose of improving Autonomous Underwater Vehicles (AUVs) used in survey missions. Further information can be found here: http://www.irs.uji.es/twinbot/twinbot.html#

Tasked with solving the object detection problems the robots will face, a project to implement an automatic 3D segmentation of underwater objects (for example, underwater pipes), i.e. identifying and labelling objects in a given surrounding, using Deep Learning (DL), is created.
This project comprises of:
- ROS for stereo image obtention and calibration.
- Python for disparity image and 3D Point Cloud generation.
- 3D U-Net for training = Automatic image segmentation.
# Project tasks

- Obtain disparity images and point clouds from stereo images.
Involves identifying identical features in both left and right stereo images, estimating the depth of the objects in the images and reproducing the 3D structures.
- Obtain ground truth images from disparity images.
With the help of an image editor, the pipes are identified and classified. This is, basically, a manual segmentation of the images.
- Obtain cubic volumes.
We calculate the missing depth values of the 2D stereo images to create a 3D cubic volume for training.
 
 # 
