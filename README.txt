This software implements MATLAB function to crop tetragons of an image and resize them to the standard size (on a GPU):
a wrapper on top of NVIDIA Performance Primitives (https://developer.nvidia.com/NPP), function nppiWarpPerspectiveQuad_32f_P3R
The NPP library is included in the standard CUDA package.

If you are using this code, please, consider citing the following paper:
Tuan-Hung Vu, Anton Osokin, Ivan Laptev. Context-aware CNNs for person head detection.
In proceedings of International Conference on Computer Vision (ICCV), 2015.

The full detection code and our data can be found on the project page: http://www.di.ens.fr/willow/research/headdetection

Anton Osokin, (firstname.lastname@gmail.com)
November, 2015
https://github.com/aosokin/cropTetragonsMex

PACKAGE
-----------------------------

./cropTetragonsMex.cu - the source code

./build_cropTetragonsMex.m - the build script

./cropTetragonsMex.m - the description of the implemented function

./example_cropTetragonsMex.m - the example of usage

USING THE CODE
-----------------------------

0) Install MATLAB, the supported compiler, and the appropriate version of CUDA

1) Run build_cropTetragonsMex.m

2) Run example_cropTetragonsMex.m to check if the code works

The code was tested under 
- ubuntu-12.04-x64 using MATLAB R2014b, gcc-4.6.3, cuda-6.5

This code was written to be used together with MatConvNet (http://www.vlfeat.org/matconvnet/) and in theory should work if MatConvNet works.
If you face compilation problems the MatConvNet compilation script (vl_compilenn.m) might be of some help.

OTHER PACKAGES
-----------------------------
If you want crop rectangles and not general tetragons see cropRectangleMex.m from https://github.com/aosokin/cropRectanglesMex
