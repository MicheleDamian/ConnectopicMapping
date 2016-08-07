TR = 5;
FILENAME = '/Users/michele/Documents/Workspaces/UpWork/Morgan_Hough/fmri/fmri_fluency/word_generation.txt';

% Read file
input = csvread(FILENAME);

% Expand input
for i = 1:length(input) 
    start = input(i, 1) * TR; 
    stop = input(i, 2) * TR-1; 
    waveform(1, start:start+stop) = input(i, 3); 
end

% Build kernel for convolution
x = 0:150; x = x / TR;
sigma1 = 2.449; sigma2 = 4;
lag1 = 6; lag2 = 16; 
ratio = 6;
kernel = gampdf(x, lag1^2 / sigma1^2, sigma1^2 / lag1) - gampdf(x, lag2^2 / sigma2^2, sigma2^2 / lag2)/ratio;
y_conv = conv(kernel, waveform);

% Add low frequency linear signal
%y_conv = y_conv + (1:numel(y_conv)) / 1000;
% Add low frequency sinusoidal signal
%y_conv = y_conv + sin((1:numel(y_conv))/300);

sigmas = [10.0];
g_color = jet(numel(sigmas)+1);
g_legend = {'Stimulous'};

figure(1);
plot(0:length(y_conv)-1, y_conv, 'Color', g_color(1, :));
hold on;

for i = 1:numel(sigmas)
    % Build highpass filter
    y_filt = feat_bandpass(y_conv, TR, 0, sigmas(i), 1);
    % Plot stimulus high passed
    plot(0:length(y_filt)-1, y_filt, 'Color', g_color(1+i, :)); hold on;
    g_legend{1+i} = sprintf('Highpass %d', sigmas(i));
end

grid on;
title('Stimulus');
legend(g_legend);

