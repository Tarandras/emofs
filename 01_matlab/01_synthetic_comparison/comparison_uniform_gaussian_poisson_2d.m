clear all 
close all 
addpath('..')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Experiment Settings
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PDF configuration
x_min = 20; x_max = 30;
y_min = 0; y_max = 10;
x_min_g = 0; x_max_g = -10;
y_min_g = 0; y_max_g = -10; 
p_lambda_x = 5;
p_lambda_y = 5;
mu_x = (x_min_g + x_max_g)/2; % Gaussian parameters
mu_y = (y_min_g + y_max_g)/2;
sigma_x = (x_max - x_min) / 2;
sigma_y = (y_max - y_min) / 2;
sampleSize = 1500; % total samples per distribution
HalfPeriod_L = [ x_max+(x_max - x_min)*3 y_max+(y_max - y_min)*3 ];
Mixture_N = 20; % total number of mixture components
Trimming_Factor = 20; 
Convergence_Factor = 0.0000001;
Convergence_Debug = true; 
% view settings
my_xlim = 60; 
my_ylim = 60; 
my_zlim = 0.011;
my_view = [348 21];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load Python Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
parentDir = fullfile(pwd, '..');
if count(py.sys.path, parentDir) == 0
    insert(py.sys.path, int32(0), parentDir);
end
py.importlib.invalidate_caches();

mod = py.importlib.import_module('flow_models');
py.importlib.reload(mod);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sample generation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng('shuffle');
% Uniform samples
x_samples_uniform = x_min + (x_max - x_min) * rand( sampleSize, 1 );
y_samples_uniform = y_min + (y_max - y_min) * rand( sampleSize, 1 );
% Gaussian samples
x_samples_gaussian = mu_x + sigma_x * randn(sampleSize, 1);
y_samples_gaussian = mu_y + sigma_y * randn(sampleSize, 1);
% Poisson samples
x_samples_poisson = poissrnd(p_lambda_x, sampleSize, 1);
y_samples_poisson = poissrnd(p_lambda_y, sampleSize, 1);
% Input concatenation & shuffle
x_input = [x_samples_gaussian x_samples_uniform x_samples_poisson];
y_input = [y_samples_gaussian y_samples_uniform y_samples_poisson];
inputArray = [x_input(:), y_input(:)];
idx = randperm(size(inputArray,1));
inputArray = inputArray(idx, :);
% plot input in scattered form 
figure;
scatter(x_input, y_input);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EMoFS Estimation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[pdf_index, Estimated_PDF] = emofs( ...
    inputArray, HalfPeriod_L, Mixture_N, Trimming_Factor, ...
    Convergence_Factor, Convergence_Debug); 
% Generate X,Y,Z indices from pdf_index
MaxPeriod = max(HalfPeriod_L) * 2 + 1;
x_axis = pdf_index(:,1);
y_axis = pdf_index(:,2);
z_axis = Estimated_PDF;
% Reshape into grid form
X = reshape(x_axis, MaxPeriod, MaxPeriod);
Y = reshape(y_axis, MaxPeriod, MaxPeriod);
Z_emofs = reshape(z_axis, MaxPeriod, MaxPeriod);
% Plot the estimated pdf
figure;
surf(X, Y, Z_emofs); 
xlabel('x (in)');
ylabel('y (in)');
zlabel('Estimated PDF by EMoFS');
set(gca,'ydir','reverse');
% Show in 2D contour
figure;
contour(X, Y, Z_emofs, Mixture_N, 'k');  % 2D contour

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Original PDF and its illustration
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Uniform
pdf_val = 1 / ((x_max - x_min) * (y_max - y_min ));
Z_orig_uniform = zeros(size(X));
inside = (X >= x_min & X <= x_max) & (Y >= y_min & Y <= y_max);
Z_orig_uniform(inside) = pdf_val;
% Gaussian
Z_orig_gaussian = (1 / (2*pi*sigma_x*sigma_y)) * ...
      exp(-((X - mu_x).^2 / (2*sigma_x^2)+...
      (Y - mu_y).^2 / (2*sigma_y^2)));
% Poisson
Z_orig_poisson = poisspdf(X, p_lambda_x) .* poisspdf(Y, p_lambda_y);
% Concatenation & Normalization
Z_orig = Z_orig_uniform + Z_orig_gaussian + Z_orig_poisson;
Z_orig = Z_orig / sum(Z_orig(:));
% Plot
figure;
surf(X, Y, Z_orig); 
xlabel('x (in)');
ylabel('y (in)');
zlabel('Original PDF');
set(gca,'ydir','reverse');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% IGMM Estimation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
options = statset('MaxIter',1000, 'Display','final');
gmdist = fitgmdist(inputArray, Mixture_N,'Options',options); % Mixture_N components
grid_points = [x_axis, y_axis];
f = pdf(gmdist, grid_points);
Z_igmm = reshape(f, MaxPeriod, MaxPeriod);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Masked Autoregressive Flow Estimation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model_type = 'maf';  % 'maf' | 'nsf' | 'realnvp'
result = mod.run_flow( ...
    py.numpy.array(inputArray), ...
    py.numpy.array(grid_points), ...
    model_type);
p = double(result);
Z_maf = reshape(p, MaxPeriod, MaxPeriod);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Neural Spline Flow Estimation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model_type = 'nsf';  % 'maf' | 'nsf' | 'realnvp'
result = mod.run_flow( ...
    py.numpy.array(inputArray), ...
    py.numpy.array(grid_points), ...
    model_type);
p = double(result);
Z_nsf = reshape(p, MaxPeriod, MaxPeriod);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Real-valued Non-Volume Preserving flows
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model_type = 'realnvp';  % 'maf' | 'nsf' | 'realnvp'
result = mod.run_flow( ...
    py.numpy.array(inputArray), ...
    py.numpy.array(grid_points), ...
    model_type);
p = double(result);
Z_realnvp = reshape(p, MaxPeriod, MaxPeriod);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Wavelet Density Estimator
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Grid histogram (raw density)
bins = MaxPeriod+1;
edges1 = linspace(min(inputArray(:,1)), max(inputArray(:,1)), bins);
edges2 = linspace(min(inputArray(:,2)), max(inputArray(:,2)), bins);
H = histcounts2(inputArray(:,1), inputArray(:,2), edges1, edges2);
H = H / sum(H(:)); % normalize
% Wavelet decomposition
wavelet = 'db2';
level = 2;
[C,S] = wavedec2(H, level, wavelet);
% Thresholding (denoising)
sigma = median(abs(C)) / 0.6745;
threshold = sigma * sqrt(2*log(numel(H)));
C_filt = wthresh(C, 's', threshold);
% Reconstruction
H_smooth = waverec2(C_filt, S, wavelet);
% Ensure positivity
H_smooth = max(H_smooth, 0);
H_smooth = H_smooth / sum(H_smooth(:));
Z_wavelet = reshape(H_smooth, MaxPeriod, MaxPeriod);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% KDE 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[f, xi, bw] = ksdensity(inputArray, grid_points); 
Z_kde = reshape(f, MaxPeriod, MaxPeriod);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Draw Wavelet and Original Together
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawTogether(X, Y, my_xlim, my_ylim, my_zlim, my_view, Z_orig, ...
    Z_wavelet, 'Original PDF vs Wavelet', 'Estimation by Wavelet')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Draw RealNPV and Original Together
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawTogether(X, Y, my_xlim, my_ylim, my_zlim, my_view, Z_orig, ...
    Z_realnvp, 'Original PDF vs RealNVP', 'Estimation by RealNVP')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Draw NSF and Original Together
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawTogether(X, Y, my_xlim, my_ylim, my_zlim, my_view, Z_orig, Z_nsf, ...
    'Original PDF vs NSF', 'Estimation by NSF')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Draw MAF and Original Together
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawTogether(X, Y, my_xlim, my_ylim, my_zlim, my_view, Z_orig, Z_maf, ...
    'Original PDF vs MAF', 'Estimation by MAF')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Draw KDE and Original Together
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawTogether(X, Y, my_xlim, my_ylim, my_zlim, my_view, Z_orig, Z_emofs, ...
    'Original PDF vs KDE', 'Estimation by KDE')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Draw IGMM and Original Together
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawTogether(X, Y, my_xlim, my_ylim, my_zlim, my_view, Z_orig, Z_igmm, ...
    'Original PDF vs IGMM', 'Estimation by IGMM')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Draw EMoFS and Original Together
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawTogether(X, Y, my_xlim, my_ylim, my_zlim, my_view, Z_orig, Z_emofs, ...
    'Original PDF vs EMoFS', 'Estimation by EMoFS')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% RMS calculation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
error_diff = Z_orig - Z_emofs;
error_rms = sqrt(sum(error_diff.^2, "all")); 
fprintf("RSS EMoFS:  %f\n", error_rms);
error_diff = Z_orig - Z_igmm;
error_rms = sqrt(sum(error_diff.^2, "all")); 
fprintf("RSS IGMM:  %f\n", error_rms);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% colormapToTruecolor function description
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function tc = colormapToTruecolor(map,ZData)
    % map is a n-by-3 colormap matrix
    % ZData is a k-by-w matrix of ZData (or CData, I suppose)
    % tc is a kxwx3 Truecolor array based on map and ZData values.
    tcIdx = round(rescale(ZData,1,height(map)));
    tc = reshape(map(tcIdx,:),[size(ZData),3]);
end