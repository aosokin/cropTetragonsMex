
#include "mex.h"
#include "gpu/mxGPUArray.h"

#include <npp.h>

#include <math.h> 

#define MATLAB_ASSERT(expr,msg) if (!(expr)) { mexErrMsgTxt(msg);}

#if !defined(MX_API_VER) || MX_API_VER < 0x07030000
typedef size_t mwSize;
typedef size_t mwIndex;
#endif

void mexFunction(int nlhs, mxArray *plhs[], 
    int nrhs, const mxArray *prhs[])
{
	MATLAB_ASSERT( nrhs == 3, "cropTetragonsMex: Wrong number of input parameters: expected 3");
    MATLAB_ASSERT( nlhs == 1, "cropTetragonsMex: Wrong number of output arguments: expected 1");
	
	// Fix input parameter order:
	const mxArray *imInPtr = (nrhs >= 0) ? prhs[0] : NULL; // image
	const mxArray *tetragonInPtr = (nrhs >= 1) ? prhs[1] : NULL; // tetragons
	const mxArray *szInPtr = (nrhs >= 2) ? prhs[2] : NULL; // output image size
	
	// Fix output parameter order:
	mxArray **cropsOutPtr = (nlhs >= 1) ? &plhs[0] : NULL; // croped and resized patches
	
	// Get the image
	MATLAB_ASSERT(mxGetNumberOfDimensions(imInPtr) == 3, "cropTetragonsMex: the image is not 3-dimensional");
	MATLAB_ASSERT(mxGetClassID(imInPtr) == mxSINGLE_CLASS, "cropTetragonsMex: the image should be of type SINGLE");
	MATLAB_ASSERT(mxGetPi(imInPtr) == NULL, "cropTetragonsMex: image should not be complex");

    const mwSize* dimensions = mxGetDimensions(imInPtr);
	mwSize imageHeight = dimensions[0];
	mwSize imageWidth = dimensions[1];
	mwSize numChannels = dimensions[2];
	MATLAB_ASSERT(numChannels == 3, "cropTetragonsMex: image should contain 3 channels");

	float* imageData = (float*) mxGetData(imInPtr);

	// get tetragons
	MATLAB_ASSERT(mxGetNumberOfDimensions(tetragonInPtr) == 2, "cropTetragonsMex: <tetragons> input is not 2-dimensional");
	MATLAB_ASSERT(mxGetClassID(tetragonInPtr) == mxDOUBLE_CLASS, "cropTetragonsMex: <tetragons> input is not of type double");
	MATLAB_ASSERT(mxGetPi(tetragonInPtr) == NULL, "cropTetragonsMex: <tetragons> input should not be complex");
	MATLAB_ASSERT(mxGetN(tetragonInPtr) == 8, "cropTetragonsMex: <tetragons> input should be of size #tetragons x 8");
	
	mwSize numTetragon = mxGetM(tetragonInPtr);
	double* tetragonData = (double*) mxGetData(tetragonInPtr); // y1, x1, y2, x2, y3, x3, y4, x4

	// get output size
	MATLAB_ASSERT(mxGetNumberOfElements(szInPtr) == 2, "cropTetragonsMex: <outputSize> input should contain 2 numbers");
	MATLAB_ASSERT(mxGetClassID(szInPtr) == mxDOUBLE_CLASS, "cropTetragonsMex: <outputSize> input is not of type double");
	MATLAB_ASSERT(mxGetPi(szInPtr) == NULL, "cropTetragonsMex: <outputSize> input should not be complex");
	
	double* outputSizeData = (double*) mxGetData(szInPtr);
	int targetHeight = (int) (outputSizeData[0] + 0.5);
	int targetWidth = (int) (outputSizeData[1] + 0.5);

	// initialize GPU
	mxInitGPU();

	// copy image to the GPU
	mxGPUArray const *inputImage;
    float const *d_inputImage;
	inputImage = mxGPUCreateFromMxArray(imInPtr);
	d_inputImage = (float const *)(mxGPUGetDataReadOnly(inputImage));

	// allocate memory for the output
    mxGPUArray *outputData;
    float *d_outputData;
	const mwSize outputDimensions[4] = { targetHeight, targetWidth, numChannels, numTetragon };
	outputData = mxGPUCreateGPUArray(4, outputDimensions, mxSINGLE_CLASS, mxREAL, MX_GPU_INITIALIZE_VALUES) ; //MX_GPU_DO_NOT_INITIALIZE);
	d_outputData = (float *)(mxGPUGetData(outputData));

	// initialize some cropping arguments
	NppiSize nppiImageSize = {};
	nppiImageSize.width = imageHeight; // CAUTION: NPPI thinks that the image is transposed 
	nppiImageSize.height = imageWidth;

	int channelValueSize = sizeof(float);
	int imageStep = imageHeight * channelValueSize;
	int targetStep = targetHeight * channelValueSize;

    NppiRect targetRect = {};
    targetRect.x = 0;
    targetRect.y = 0;
    targetRect.width = targetHeight;
    targetRect.height = targetWidth;

    double aDstQuad[4][2] = { {0.0, 0.0}, {targetHeight - 1.0, 0.0}, {targetHeight - 1.0, targetWidth - 1.0}, {0.0, targetWidth - 1.0} };
	
	// the main loop over bounding boxes
	for(int iBb = 0; iBb < numTetragon; ++iBb) {

		double y1 = tetragonData[ iBb ] - 1;
		double x1 = tetragonData[ iBb + numTetragon ] - 1;
		double y2 = tetragonData[ iBb + numTetragon * 2 ] - 1;
		double x2 = tetragonData[ iBb + numTetragon * 3 ] - 1;
		double y3 = tetragonData[ iBb + numTetragon * 4 ] - 1;
		double x3 = tetragonData[ iBb + numTetragon * 5 ] - 1;
		double y4 = tetragonData[ iBb + numTetragon * 6 ] - 1;
		double x4 = tetragonData[ iBb + numTetragon * 7 ] - 1;

		NppiRect sourceRect = {};
    	sourceRect.x = (int) floor(min( min(y1, y2), min(y3, y4) ));
    	sourceRect.y = (int) floor(min( min(x1, x2), min(x3, x4) ));
    	sourceRect.width =  (int) ceil( max( max(y1, y2), max(y3, y4) ) - min( min(y1, y2), min(y3, y4) ) + 1);
    	sourceRect.height = (int) ceil( max( max(x1, x2), max(x3, x4) ) - min( min(x1, x2), min(x3, x4) ) + 1);

    	// adjust bounding box bounds if it is outside of the image
    	if (sourceRect.x < 0) {
    		sourceRect.width = sourceRect.width + sourceRect.x;
    		sourceRect.x = 0.0;
    	}
    	if (sourceRect.y < 0) {
    		sourceRect.height = sourceRect.height + sourceRect.y;
    		sourceRect.y = 0.0;
    	}
    	if (sourceRect.width > imageHeight  - sourceRect.x + 1) {
    		sourceRect.width = imageHeight  - sourceRect.x + 1;
    	}
    	if (sourceRect.height > imageWidth  - sourceRect.y + 1) {
    		sourceRect.height = imageWidth  - sourceRect.y + 1;
    	}

		// double aSrcQuad[4][2] = { {y1 + 0.5, x1 + 0.5}, {y4 + 0.5, x4 + 0.5}, {y3 + 0.5, x3 + 0.5}, {y2 + 0.5, x2 + 0.5} };
		double aSrcQuad[4][2] = { {y1, x1}, {y4, x4}, {y3, x3}, {y2, x2} };

    	float *curOutput = d_outputData + numChannels * targetHeight * targetWidth * iBb;
		const float *pSrc[3] = { d_inputImage, d_inputImage + imageHeight * imageWidth, d_inputImage + 2 * imageHeight * imageWidth};
		float *pDst[3] = { curOutput, curOutput + targetHeight * targetWidth, curOutput + 2 * targetHeight * targetWidth};

        // When NPP_CHECK_NPP catches an error it throws an exception
        // If the exception is not caught, we can get a memory leak on a GPU
        try{
            NppStatus exitCode = nppiWarpPerspectiveQuad_32f_P3R (
                pSrc, // const Npp32f ∗ pSrc[3], 
                nppiImageSize, // NppiSize oSrcSize, 
                imageStep, // int nSrcStep, 
                sourceRect, // NppiRect oSrcROI, 
                aSrcQuad, // const double aSrcQuad[4][2], 
                pDst, // Npp32f ∗ pDst[3], 
                targetStep, // int nDstStep, 
                targetRect, // NppiRect oDstROI, 
                aDstQuad, // const double aDstQuad[4][2], 
                NPPI_INTER_CUBIC //int eInterpolation
                );
            if (exitCode != NPP_SUCCESS) {
                mexPrintf("nppiWarpPerspectiveQuad_32f_P3R returns exit code %d, see http://cseweb.ucsd.edu/classes/wi15/cse262-a/static/cuda-5.5-doc/pdf/NPP_Library.pdf for the description of exit code.\n", exitCode);
                MATLAB_ASSERT(exitCode == NPP_SUCCESS, "cropTetragonsMex: nppiWarpPerspectiveQuad_32f_P3R returns bad exit code");
            }
        } catch (...) {
            // free GPU memory
            mxGPUDestroyGPUArray(outputData);
            mxGPUDestroyGPUArray(inputImage);
            throw;
        }
}
	
	*cropsOutPtr = mxGPUCreateMxArrayOnGPU(outputData);

	// do not forget to free GPU memory
	mxGPUDestroyGPUArray(outputData);
	mxGPUDestroyGPUArray(inputImage);
}
