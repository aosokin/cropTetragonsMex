function build_cropTetragonsMex( cudaRoot )
%build_cropTetragonsMex builds package cropTetragonsMex
%
% INPUT:
%   cudaRoot - path to the CUDA installation

% Anton Osokin, firstname.lastname@gmail.com, November 2015

if ~exist('cudaRoot', 'var')
    cudaRoot = '/usr/cuda-7.0' ;
end
nvccPath = fullfile(cudaRoot, 'bin', 'nvcc');
if ~exist(nvccPath, 'file')
    error('NVCC compiler was not found!');
end

root = fileparts( mfilename('fullpath') );

% compiling
compileCmd = [ '"', nvccPath, '"', ...
        ' -c ', fullfile(root,'cropTetragonsMex.cu'), ...
        ' -DNDEBUG -DENABLE_GPU', ...
        ' -I"', fullfile( matlabroot, 'extern', 'include'), '"', ...
        ' -I"', fullfile( matlabroot, 'toolbox', 'distcomp', 'gpu', 'extern', 'include'), '"', ...
        ' -I"', fullfile( cudaRoot, 'include'), '"', ...
        ' -Xcompiler', ' -fPIC', ...
        ' -o "', fullfile(root,'cropTetragonsMex.o'), '"'];
system( compileCmd );

% linking
mopts = {'-outdir', root, ...
         '-output', 'cropTetragonsMex', ...
         ['-L', fullfile(cudaRoot, 'lib64')], ...
         '-lcudart', '-lnppi', '-lnppc', '-lmwgpu', ...
         fullfile(root,'cropTetragonsMex.o') };
mex(mopts{:}) ;

delete( fullfile(root,'cropTetragonsMex.o') );
