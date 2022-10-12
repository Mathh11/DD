function S = L0Restoration(Im, kernel, lambda, kappa)
%%
% Image restoration with L0 prior
% The objective function: 
% S^* = argmin ||I*k - B||^2 + lambda |\nabla I|_0
%% Input:
% @Im: Blurred image
% @kernel: blur kernel
% @lambda: weight for the L0 prior
% @kappa: Update ratio in the ADM
%% Output:
% @S: Latent image

if ~exist('kappa','var')
    kappa = 2.0;
end

%% pad image
H = size(Im,1);    W = size(Im,2);
Im = wrap_boundary_liu(Im, opt_fft_size([H W]+size(kernel)-1));

%%
S = Im;
betamax = 1e5;
fx = [1, -1];
fy = [1; -1];
[N,M,D] = size(Im);
sizeI2D = [N,M];
otfFx = psf2otf(fx,sizeI2D);
otfFy = psf2otf(fy,sizeI2D);

%%
KER = psf2otf(kernel,sizeI2D);
Den_KER = abs(KER).^2;

%%
Denormin2 = abs(otfFx).^2 + abs(otfFy ).^2;
Normin1 = conj(KER).*fft2(S);

%% 
beta = 2*lambda;
while beta < betamax
    Denormin   = Den_KER + beta*Denormin2;
    h = [diff(S,1,2), S(:,1,:) - S(:,end,:)];
    v = [diff(S,1,1); S(1,:,:) - S(end,:,:)];
    t = (h.^2+v.^2)<lambda/beta;
    h(t)=0; v(t)=0;
    Normin2 = [h(:,end,:) - h(:, 1,:), -diff(h,1,2)];
    Normin2 = Normin2 + [v(end,:,:) - v(1, :,:); -diff(v,1,1)];
    FS = (Normin1 + beta*fft2(Normin2))./Denormin;
    S = real(ifft2(FS));
    beta = beta*kappa;
end
S = S(1:H, 1:W, :);
end
