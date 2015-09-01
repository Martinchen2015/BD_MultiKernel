hFig = figure(1);
set(hFig, 'Position', [1000 200 720 720]);

imagesc(abs(Y)); axis off;
export_fig 'noisyrec_A.pdf' -transparent

imagesc(abs(Aout)); axis off;
export_fig 'noisyrec_B.pdf' -transparent

imagesc(abs(Xout)); axis off;
export_fig 'noisyrec_C.pdf' -transparent

imagesc(abs(cconvfft2(Aout,Xout))); axis off;
export_fig 'noisyrec_D.pdf' -transparent
