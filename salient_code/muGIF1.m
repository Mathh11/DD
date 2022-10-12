function [S] = muGIF1(T,alpha_t,maxiter,mode) 
    
    T0 = im2double(T);
    t = T0;
    epst = 0.01;
    
    if mode == 2   % dynamic only
        for i = 1 :maxiter
            [wtx, wty] = computeTextureWeights(t,epst);
            wx = wtx.^2;
            wy = wty.^2;
            t = solveLinearEquation(T0, wx, wy, alpha_t);
        end
   end
    S = t;   
end

function [retx, rety] = computeTextureWeights(fin,vareps_s)
   fx = diff(fin,1,2);
   fx = padarray(fx, [0 1], 'post');
   fy = diff(fin,1,1);
   fy = padarray(fy, [1 0], 'post');

   retx = max(max(abs(fx),[],3),vareps_s).^(-1);  %%对应公式5计算Q
   rety = max(max(abs(fy),[],3),vareps_s).^(-1);

   retx(:,end) = 0;
   rety(end,:) = 0;
end

function OUT = solveLinearEquation(IN, wx, wy, lambda)
% WLS
    [r,c,ch] = size(IN);
    k = r*c;
    dx = -lambda*wx(:);
    dy = -lambda*wy(:);
    B(:,1) = dx;
    B(:,2) = dy;
    d = [-r,-1];
    A = spdiags(B,d,k,k);
    e = dx;
    w = padarray(dx, r, 'pre'); w = w(1:end-r);
    s = dy;
    n = padarray(dy, 1, 'pre'); n = n(1:end-1);
    D = 1-(e+w+s+n);
    A = A + A' + spdiags(D, 0, k, k); 
    L = ichol(A,struct('michol','on'));    
    
     OUT = IN;
        for ii=1:ch
            tin = IN(:,:,ii);
            [tout,~] = pcg(A, tin(:),.01,max(min(ceil(lambda*100),40),10), L, L'); 
            OUT(:,:,ii) = reshape(tout, r, c);
        end    
end