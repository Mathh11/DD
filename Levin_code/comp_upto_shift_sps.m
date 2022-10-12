function [ssde,psnrs,ssims,tI1]=comp_upto_shift_sps(I1,I2)
%function [ssde,tI1]=comp_upto_shift(I1,I2)
%  compute sum of square differences between two images, after
%  finding the best shift between them. need to account for shift
%  because the kernel reconstruction is shift invariant- a small 
%  shift of the image and kernel will not effect the likelihood score.
%Input:
%I1,I2-images to compare
%Output:
%ssde-sum of square differences
%tI1-image I1 at best shift toward I2
%Writen by: Anat Levin, anat.levin@weizmann.ac.il (c)

    [N1,N2]=size(I1);
    maxshift=5;
    shifts=[-5:0.25:5];
    I2=I2(16:end-15,16:end-15);
    I1=I1(16-maxshift:end-15+maxshift,16-maxshift:end-15+maxshift);
    [N1,N2]=size(I2);
    [gx,gy]=meshgrid([1-maxshift:N2+maxshift],[1-maxshift:N1+maxshift]);

    [gx0,gy0]=meshgrid([1:N2],[1:N1]);

    for i=1:length(shifts)
       for j=1:length(shifts)
         gxn=gx0+shifts(i);
         gyn=gy0+shifts(j);
         tI1=interp2(gx,gy,I1,gxn,gyn);
         ssdem(i,j)=sum(sum((tI1-I2).^2));   
       end
    end

    ssde=min(ssdem(:));
    [i,j]=find(ssdem==ssde);

    gxn=gx0+shifts(i);
    gyn=gy0+shifts(j);
    tI1=interp2(gx,gy,I1,gxn,gyn);
    psnrs = psnr(255*tI1,255*I2);
    ssims = ssim(255*tI1,255*I2);
end
function [ PSNR, MSE ] = psnr( f1,f2 )
%PSNR Summary of this function goes here
%   Detailed explanation goes here
% % MSE = E( (img－Eimg)^2 ) 
% %     = SUM((img-Eimg)^2)/(M*N);
% function ERMS = erms(f1, f2)
% %计算f1和f2均方根误差
% e = double(f1) - double(f2);
% [m, n] = size(e);
% ERMS = sqrt(sum(e.^2)/(m*n));
% % PSNR=10log10(M*N/MSE); 
% function PSNR = psnr(f1, f2)
%计算两幅图像的峰值信噪比
    k=1;
    if max(f1(:))>2
        k = 8;
    end
    %k为图像是表示地个像素点所用的二进制位数，即位深。
    fmax = 2.^k - 1;
    a = fmax.^2;
    e = double(f1) - double(f2);
    [m, n] = size(e);
    MSE=sum(sum(e.^2))/(m*n);
    PSNR = 10*log10(a/MSE);
end
function [mssim, ssim_map] = ssim(img1, img2, K, window, L)

% ========================================================================
% SSIM Index with automatic downsampling, Version 1.0
% Copyright(c) 2009 Zhou Wang
% All Rights Reserved.
%
% ----------------------------------------------------------------------
% Permission to use, copy, or modify this software and its documentation
% for educational and research purposes only and without fee is hereby
% granted, provided that this copyright notice and the original authors'
% names appear on all copies and supporting documentation. This program
% shall not be used, rewritten, or adapted as the basis of a commercial
% software or hardware product without first obtaining permission of the
% authors. The authors make no representations about the suitability of
% this software for any purpose. It is provided "as is" without express
% or implied warranty.
%----------------------------------------------------------------------
%
% This is an implementation of the algorithm for calculating the
% Structural SIMilarity (SSIM) index between two images
%
% Please refer to the following paper and the website with suggested usage
%
% Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image
% quality assessment: From error visibility to structural similarity,"
% IEEE Transactios on Image Processing, vol. 13, no. 4, pp. 600-612,
% Apr. 2004.
%
% http://www.ece.uwaterloo.ca/~z70wang/research/ssim/
%
% Note: This program is different from ssim_index.m, where no automatic
% downsampling is performed. (downsampling was done in the above paper
% and was described as suggested usage in the above website.)
%
% Kindly report any suggestions or corrections to zhouwang@ieee.org
%
%----------------------------------------------------------------------
%
%Input : (1) img1: the first image being compared
%        (2) img2: the second image being compared
%        (3) K: constants in the SSIM index formula (see the above
%            reference). defualt value: K = [0.01 0.03]
%        (4) window: local window for statistics (see the above
%            reference). default widnow is Gaussian given by
%            window = fspecial('gaussian', 11, 1.5);
%        (5) L: dynamic range of the images. default: L = 255
%
%Output: (1) mssim: the mean SSIM index value between 2 images.
%            If one of the images being compared is regarded as 
%            perfect quality, then mssim can be considered as the
%            quality measure of the other image.
%            If img1 = img2, then mssim = 1.
%        (2) ssim_map: the SSIM index map of the test image. The map
%            has a smaller size than the input images. The actual size
%            depends on the window size and the downsampling factor.
%
%Basic Usage:
%   Given 2 test images img1 and img2, whose dynamic range is 0-255
%
%   [mssim, ssim_map] = ssim(img1, img2);
%
%Advanced Usage:
%   User defined parameters. For example
%
%   K = [0.05 0.05];
%   window = ones(8);
%   L = 100;
%   [mssim, ssim_map] = ssim(img1, img2, K, window, L);
%
%Visualize the results:
%
%   mssim                        %Gives the mssim value
%   imshow(max(0, ssim_map).^4)  %Shows the SSIM index map
%========================================================================
    if (nargin < 2 || nargin > 5)
       mssim = -Inf;
       ssim_map = -Inf;
       return;
    end

    if (size(img1) ~= size(img2))
       mssim = -Inf;
       ssim_map = -Inf;
       return;
    end

    [M N] = size(img1);

    if (nargin == 2)
       if ((M < 11) || (N < 11))
           mssim = -Inf;
           ssim_map = -Inf;
          return
       end
       window = fspecial('gaussian', 11, 1.5);	%
       K(1) = 0.01;					% default settings
       K(2) = 0.03;					%
       L = 255;                                     %
    end

    if (nargin == 3)
       if ((M < 11) || (N < 11))
           mssim = -Inf;
           ssim_map = -Inf;
          return
       end
       window = fspecial('gaussian', 11, 1.5);
       L = 255;
       if (length(K) == 2)
          if (K(1) < 0 || K(2) < 0)
               mssim = -Inf;
            ssim_map = -Inf;
            return;
          end
       else
           mssim = -Inf;
        ssim_map = -Inf;
           return;
       end
    end

    if (nargin == 4)
       [H W] = size(window);
       if ((H*W) < 4 || (H > M) || (W > N))
           mssim = -Inf;
           ssim_map = -Inf;
          return
       end
       L = 255;
       if (length(K) == 2)
          if (K(1) < 0 || K(2) < 0)
               mssim = -Inf;
            ssim_map = -Inf;
            return;
          end
       else
           mssim = -Inf;
        ssim_map = -Inf;
           return;
       end
    end

    if (nargin == 5)
       [H W] = size(window);
       if ((H*W) < 4 || (H > M) || (W > N))
           mssim = -Inf;
           ssim_map = -Inf;
          return
       end
       if (length(K) == 2)
          if (K(1) < 0 || K(2) < 0)
               mssim = -Inf;
            ssim_map = -Inf;
            return;
          end
       else
           mssim = -Inf;
        ssim_map = -Inf;
           return;
       end
    end


    img1 = double(img1);
    img2 = double(img2);

    % automatic downsampling
    f = max(1,round(min(M,N)/256));
    %downsampling by f
    %use a simple low-pass filter 
    if(f>1)
        lpf = ones(f,f);
        lpf = lpf/sum(lpf(:));
        img1 = imfilter(img1,lpf,'symmetric','same');
        img2 = imfilter(img2,lpf,'symmetric','same');

        img1 = img1(1:f:end,1:f:end);
        img2 = img2(1:f:end,1:f:end);
    end

    C1 = (K(1)*L)^2;
    C2 = (K(2)*L)^2;
    window = window/sum(sum(window));

    mu1   = filter2(window, img1, 'valid');
    mu2   = filter2(window, img2, 'valid');
    mu1_sq = mu1.*mu1;
    mu2_sq = mu2.*mu2;
    mu1_mu2 = mu1.*mu2;
    sigma1_sq = filter2(window, img1.*img1, 'valid') - mu1_sq;
    sigma2_sq = filter2(window, img2.*img2, 'valid') - mu2_sq;
    sigma12 = filter2(window, img1.*img2, 'valid') - mu1_mu2;

    if (C1 > 0 && C2 > 0)
       ssim_map = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))./((mu1_sq + mu2_sq + C1).*(sigma1_sq + sigma2_sq + C2));
    else
       numerator1 = 2*mu1_mu2 + C1;
       numerator2 = 2*sigma12 + C2;
        denominator1 = mu1_sq + mu2_sq + C1;
       denominator2 = sigma1_sq + sigma2_sq + C2;
       ssim_map = ones(size(mu1));
       index = (denominator1.*denominator2 > 0);
       ssim_map(index) = (numerator1(index).*numerator2(index))./(denominator1(index).*denominator2(index));
       index = (denominator1 ~= 0) & (denominator2 == 0);
       ssim_map(index) = numerator1(index)./denominator1(index);
    end

    mssim = mean2(ssim_map);

    return
end


