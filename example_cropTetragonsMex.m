
initialImage = im2single( imread('peppers.png') );

width = size(initialImage, 2);
height = size(initialImage, 1);

cropPositions = [ 1, 1, 1, width, height, width, height, 1; ... % full image
                  -10, 1, -10, height / 3, width / 3, height / 3, width / 3, 1 ; ... sub patch
                  1, width / 2, height / 2, width, height, width / 2, height / 2, 1 ]; % heavy transformation
crops = cropTetragonsMex( initialImage, cropPositions, [height, width] );

figure(1), imshow( initialImage );
figure(2), imshow( crops(:,:,:,1) );
figure(3), imshow( crops(:,:,:,2) );
figure(4), imshow( crops(:,:,:,3) );

