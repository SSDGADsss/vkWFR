%%前处理
clc;close all;clear
%生成条纹图测试
H = 8192;
L = 8192;
p = 6; 
f0 = 1/p;
a = 128; 
b = 128;
x1 = 1:L;
y1 = 1:H;
[X,Y] = meshgrid(x1,y1);
%添加变形相位场
u=0.0005*X;
%kk=0.3;
%v=0.01*Y+kk*peaks(H);
v=0.0005*Y;
% figure;imagesc(v);title('real_v');colorbar
ImgRef = a + b/2* cos(2* pi* f0* X)+ b/2* cos(2* pi* f0* Y);
ImgRef=ImgRef+5*randn(H,L);
figure;imshow(uint8(ImgRef));title('初始网格')
ImgDef = a + b/2* cos(2* pi* f0* X+u)+ b/2* cos(2* pi* f0* Y+v);
ImgDef=ImgDef+5*randn(H,L);
figure;imshow(uint8(ImgDef));title('变形网格')
imwrite(uint8(ImgRef),sprintf("%dx%d_1.bmp",H,L));imwrite(uint8(ImgDef),sprintf("%dx%d_2.bmp",H,L));
% 
% ImgRef=imread("D:\r.bmp");
% ImgDef=imread("D:\d.bmp");
% ImgRef = double(ImgRef);ImgDef = double(ImgDef);

% if exist('moireROI.mat','file')
% load('moireROI.mat');
% else
% imshow(ImgRef);
% ROI=round(getrect);
% save('moireROI.mat',"ROI");
% close gcf
% end
% 
% ImgRef=ImgRef(ROI(2):ROI(2)+ROI(4)-1,ROI(1):ROI(1)+ROI(3)-1);
% %figure(1);imshow(ImgRef);
% ImgRef=double(ImgRef);
% 
% ImgDef=imread("H:\ExperimentData\新建文件夹\BMP192.168.8.68-20250513-101017\20250513_000008_680000-15.bmp");
% ImgDef=ImgDef(ROI(2):ROI(2)+ROI(4)-1,ROI(1):ROI(1)+ROI(3)-1);
% %figure(1);imshow(ImgRef);
% ImgDef=double(ImgDef);

% ImgRef=imread("D:\0A最近的文件\qian_1_5\x\0_01_files\000001.bmp");
% %for i = 1:length(X)
% ImgDef=imread('img_026.bmp');
% ImgRef=ImgRef(266:340,18:280); %test
% ImgDef=ImgDef(266:340,18:280);
% ImgDef=double(ImgDef);
% ImgRef=double(ImgRef);

%% calculate phase，SM（sampling moire）

%input parameter
% p = 5;  %periodpixels
% m = 1; %Phase starting pointx；Replace if there is an error
% n = 5; %Phase starting pointy
% P = p/2; %the pitch of grid or fringe, uint:mm
% w = 10; % window of filter

% ImgRef = imread('cal_L_1_0.bmp');
% ImgDef = imread('cal_L_2_0.bmp');
% ImgRef=ImgRef(266:340,18:280); %test
% ImgDef=ImgDef(266:340,18:280);
% ImgDef=double(ImgDef);
% ImgRef=double(ImgRef);

% [phix, phiy] = SMphase(p, m, n, ImgRef, ImgDef); %得到xy方向相位
% phix= imgaussfilt(phix, w);
% phiy= imgaussfilt(phiy, w);
% figure;imagesc(phix);colorbar;axis on; title('u phase')
% figure;imagesc(phiy);colorbar;axis on; title('v phase')

%save D:\z6_00005.mat phix phiy 

% calculate displacement
% disu=P/2/pi*phix; % u dis  negative sign according to the experiment
% disv=P/2/pi*phiy; % v dis

