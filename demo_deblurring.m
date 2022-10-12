clc;
clear;
close all;
addpath(genpath('image'));
addpath(genpath('whyte_code'));
addpath(genpath('cho_code'));
addpath(genpath('salient_code'));
opts.prescale = 1; %%downsampling
opts.xk_iter = 5; %% the iterations
opts.gamma_correct = 1.0;
opts.k_thresh = 20;

alpha0 = 5e-3; 
alpha1 = 3e-3; 
alpha2 = 3e-5; 
belta_grad = 4e-3;

lambda_tv = 0.001;
lambda_l0 = 2e-4;
weight_ring = 0;
saturation = 0;

filename = 'test_image/600.png'; 
opts.kernel_size = 35;  
 
y = imread(filename);
isselect = 0; %false or true
if isselect ==1
    figure, imshow(y);
    fprintf('Please choose the area for deblurring:\n');
    h = imrect;
    position = wait(h);
    close;
    B_patch = imcrop(y,position);
    y = (B_patch);  
else
    y = y;
end
if size(y,3)==3
    yg = im2double(rgb2gray(y)); 
else
    yg = im2double(y);
end  

tic;
[kernel, interim_latent] = blind_deconv(yg, alpha0, alpha1, alpha2, belta_grad,opts);%主要定义的文章函数1
toc
y = im2double(y);

%% Final Deblur: 
if ~saturation
    %% 1. TV-L2 denoising method
    Latent = ringing_artifacts_removal(y, kernel, lambda_tv, lambda_l0, weight_ring);
else
    %% 2. Whyte's deconvolution method (For saturated images)
    Latent = whyte_deconv(y, kernel);
end
figure; imshow(Latent)



