%cropTetragonsMex crops multiple tetragons from the initial image and reshapes them to the standard output size (perspective transform)
% The operation is performed on a GPU using NVIDIA Performance Primitives (NPP) library
% cropTetragonsMex was created to prepare batches for training CNNs using MatConvNet (http://www.vlfeat.org/matconvnet/).
%
% Usage:
% crops = cropTetragonsMex( im, tetragons, outputSize);
% 	
% Inputs:
% im  - the image to crop from, should be a 3 channel image (dimension order: height, width, channels) of type single. 
%        Normalization (e.g. [0,1] or [0, 255]) is not important. The image should be stored in RAM (not GPU).
% tetragons - tetragons to crop, double[ numTetragons x 8 ], each line corresponds to one tetragon. 
%       The tetragon format is y1, x1, y2, x2, y3, x3, y4, x4, where the origin is in the top-left corner. 
%		(x1, y1) is mapped to the top-left corner (1,1) of the output, (x2, y2) - to the top-right corner (1, outputSize(2)), etc.
%       Pixels are indexed starting from 1.
%       Tetragons can be partially outside of the image. The default value for filling such areas is 0 in all the channels.
% outputSize - the target size of the resized crops, double[2 x 1]. outputSize(1) - the height, outputSize(2) - the width.
% 
% Outputs:
% crops - the cropped and resized patches, gpuArray, single[ outputSize(1), outputSize(2), numChannels = 3, numBoundingBoxes ]
%
% The function can be compiled using build_cropTetragonsMex.m. 
% example_cropTetragonsMex.m provides the example of usage.
%
% If you want crop rectangles and not general tetragons see cropRectangleMex.m from https://github.com/aosokin/cropRectanglesMex
% 
% Anton Osokin, firstname.lastname@gmail.com, November 2015

