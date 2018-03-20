<br> </br>
<b>TITLE: 3D RECONSTRUCTION USING SINGLE VIEW METROLOGY</b>
<br></br>
This repository contains code and the steps to follow for 3D Reconstruction using Single View Metrology

Steps followed to complete the project:

1. In the 1st step of the project, we took an image of a 3D box by following the 3 point
perspective imaging methodology. 

![BOX IMAGE](https://github.com/kalyanghosh/3D-Reconstruction-using-Single-View-Metrology/blob/master/Box_Image.jpg)

2. After that we used a simple annotation online tool (LABEL ME) to annotate the image.
   You can use any annotation portal to annotate the image. I used the tool <b>LABELME</b>

3. The input to the below code(SVM_Code.py) is taken from the input file (input.txt)
which is show below and also attached in the zipped folder.

4. We then wrote the code to calculate the vanishing points, the scaling factors to generate
the homography matrices and to apply the homography matrices to transform the planes.
The complete code(SVM_Code.py) is given below and also attached in the zipped
folder.

5. After getting the homography matrices from the above code,we apply the homography
matrices to each of the planes of the image to generate the texture maps for each of the
planes(XY,YZ,ZX). 
<b> PLANE XY: </b>
![XY](https://github.com/kalyanghosh/3D-Reconstruction-using-Single-View-Metrology/blob/master/xy_sc.png)
<b> PLANE XY: </b>
![XZ](https://github.com/kalyanghosh/3D-Reconstruction-using-Single-View-Metrology/blob/master/xz_sc.png)
<b> PLANE XY: </b>
![YZ](https://github.com/kalyanghosh/3D-Reconstruction-using-Single-View-Metrology/blob/master/yz_sc.png)
6. After extracting and cropping the feature maps,we generate the 3D model of the image
using VRML. The VRML code(3DModelling_Code.wrl) is as given below and also
attached in the zipped folder.

7. The .wrl file generated from the above image is opened in view3Dscene to get the 3D
image as shown below.
![3D REPRESENTATION OF BOX](https://github.com/kalyanghosh/3D-Reconstruction-using-Single-View-Metrology/blob/master/3D%20model.png)

