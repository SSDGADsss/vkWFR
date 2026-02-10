clc;
clear;
img=imread("../smallpicture/000750.bmp");
res=wft2f("wfr",img,10,-0.5,0.1,0.5,10,-0.5,0.1,0.5,6);
