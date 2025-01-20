%%
n = 64;
s = rand(1,n);
ks = fftshift(fft(s));
s_r = ifft(ifftshift(ks));
figure
plot(real(s),'LineWidth',2);hold on
plot(real(s_r),'--','LineWidth',1);
%%
n = 64;
s = rand(1,n);
b = (0:n-1) - (n/2);
p = b*pi;
s_tmp = s.*exp(-1j*p);
ks = fft(s_tmp);
s_tmp = ifft(ks);
s_r = s_tmp.*exp(1j*p);
figure
plot(real(s),'LineWidth',2);hold on
plot(real(s_r),'--','LineWidth',1);
figure
plot(abs(ks),'LineWidth',1)












