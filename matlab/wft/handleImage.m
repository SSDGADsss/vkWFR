clc;
clear;

%% real image
img=imread("../smallpicture/000750.bmp");

%rect=getrect();
rect=[238 294 100 82];
img_roi=imcrop(img,rect);
imshow(img_roi);

%% wft
tic
for ic=1:100
res=wft2f('wfr',img_roi,10,-0.5,0.1,0.5,10,-0.5,0.1,0.5,ic);
end
toc
figure;
imagesc(res.r)
% subplot(2,2,1);
% imagesc(res.wx);
% title("X Move");
% subplot(2,2,2);
% imagesc(res.wy);
% title("Y Move");
% subplot(2,2,3);
% imagesc(res.phase);
% title("Phase");
% subplot(2,2,4);
% imagesc(res.r);
% title("R");

mask=ones(size(res.r));
[p Pathmap]=unwrapping_qg_trim(complex(res.r),mask);
figure;
imagesc(Pathmap);
