clear all 
close all 
addpath('..')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Experiment Settings
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% view settings
my_xlim = 60; 
my_ylim = 60; 
my_zlim = 0.011;
my_view = [324 15];
% PDF configuration
x_min = 10; x_max = 20;
y_min = 10; y_max = 20; 
sampleSize = 1500;
HalfPeriod_L = [ x_max+(x_max - x_min)*3 y_max+(y_max - y_min)*3 ];
Mixture_N = 10;
Trimming_Factor = 20; 
Convergence_Factor = 0.000001;
Convergence_Debug = true;
% Gaussian parameters
mu_x = (x_min + x_max) / 2;
mu_y = (y_min + y_max) / 2;
sigma_x = (x_max - x_min) / 2;
sigma_y = (y_max - y_min) / 2;

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
% Generate Gaussian samples
x_samples = mu_x + sigma_x * randn(sampleSize, 1);
y_samples = mu_y + sigma_y * randn(sampleSize, 1);
% Finalize input
inputArray = [x_samples y_samples];
idx = randperm(size(inputArray,1));
inputArray = inputArray(idx, :);
% plot output
figure;
scatter(x_samples, y_samples);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EMoFS Estimation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% estimate the pdf with EMoFS
[pdf_index, Estimated_PDF] = emofs( ...
    inputArray, HalfPeriod_L, Mixture_N, Trimming_Factor, ...
    Convergence_Factor, Convergence_Debug); 

% Prepare X,Y,Z indices 
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
%surf(X, Y, Z, ...
%    'FaceColor', 'none', ...   % remove surface fill
%    'EdgeColor', 'interp');    % color edges based on Z
surf(X, Y, Z_emofs); 
xlabel('x (ph)');
ylabel('y (ph)');
zlabel('Estimated PDF by EMoFS');
set(gca,'ydir','reverse');
% 2D contour
figure;
contour(X, Y, Z_emofs, Mixture_N, 'k');  % 2D contour

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Original PDF and its illustration
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%x_orig = linspace(mu_x - 3*sigma_x, mu_x + 3*sigma_x, 100);
%y_orig = linspace(mu_y - 3*sigma_y, mu_y + 3*sigma_y, 100);
%[X_orig, Y_orig] = meshgrid(x_orig, y_orig);
Z_orig = (1 / (2*pi*sigma_x*sigma_y)) * ...
      exp(-((X - mu_x).^2 / (2*sigma_x^2)+...
      (Y - mu_y).^2 / (2*sigma_y^2)));
figure;
surf(X, Y, Z_orig); 
xlabel('x (ph)');
ylabel('y (ph)');
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
% Draw Wavelet and Original Together / normalized
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Z_wavelet = Z_wavelet / sum(Z_wavelet, "all");
drawTogether(X, Y, my_xlim, my_ylim, my_zlim, my_view, Z_orig, ...
    Z_wavelet, 'Original PDF vs Wavelet', 'Estimation by Wavelet')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Draw RealNPV and Original Together / normalized
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Z_realnvp = Z_realnvp / sum(Z_realnvp, "all");
drawTogether(X, Y, my_xlim, my_ylim, my_zlim, my_view, Z_orig, ...
    Z_realnvp, 'Original PDF vs RealNVP', 'Estimation by RealNVP')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Draw NSF and Original Together / normalized
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Z_nsf = Z_nsf / sum(Z_nsf, "all");
drawTogether(X, Y, my_xlim, my_ylim, my_zlim, my_view, Z_orig, Z_nsf, ...
    'Original PDF vs NSF', 'Estimation by NSF')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Draw MAF and Original Together / normalized
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Z_maf = Z_maf / sum(Z_maf, "all");
drawTogether(X, Y, my_xlim, my_ylim, my_zlim, my_view, Z_orig, Z_maf, ...
    'Original PDF vs MAF', 'Estimation by MAF')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Draw KDE and Original Together / normalized
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Z_kde = Z_kde / sum(Z_kde, "all");
drawTogether(X, Y, my_xlim, my_ylim, my_zlim, my_view, Z_orig, Z_kde, ...
    'Original PDF vs KDE', 'Estimation by KDE')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Draw IGMM and Original Together / normalized
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Z_igmm = Z_igmm / sum(Z_igmm, "all");
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
error_diff_emofs = Z_orig - Z_emofs;
error_rms_emofs = sqrt(sum(error_diff_emofs.^2, "all")); 
fprintf("RMS EMoFS:  %f\n", error_rms_emofs);

error_diff_igmm = Z_orig - Z_igmm;
error_rms_igmm = sqrt(sum(error_diff_igmm.^2, "all")); 
fprintf("RMS IGMM:  %f\n", error_rms_igmm);

error_diff_kde = Z_orig - Z_kde;
error_rms_kde = sqrt(sum(error_diff_kde.^2, "all")); 
fprintf("RMS KDE:  %f\n", error_rms_kde);

error_diff_wavelet = Z_orig - Z_wavelet;
error_rms_wavelet = sqrt(sum(error_diff_wavelet.^2, "all")); 
fprintf("RMS WDE:  %f\n", error_rms_wavelet);

error_diff_maf = Z_orig - Z_maf;
error_rms_maf = sqrt(sum(error_diff_maf.^2, "all")); 
fprintf("RMS MAF:  %f\n", error_rms_maf);

error_diff_nvp = Z_orig - Z_realnvp;
error_rms_nvp = sqrt(sum(error_diff_nvp.^2, "all")); 
fprintf("RMS RealNVP:  %f\n", error_rms_nvp);

error_diff_nsf = Z_orig - Z_nsf;
error_rms_nsf = sqrt(sum(error_diff_nsf.^2, "all")); 
fprintf("RMS NSF:  %f\n", error_rms_nsf);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NLL EVALUATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pdf_index_emofs = pdf_index;
pdf_index_igmm = grid_points;
pdf_index_kde = grid_points;
pdf_index_maf = grid_points;
pdf_index_nsf = grid_points;
pdf_index_realnvp = grid_points;
pdf_index_wavelet = grid_points;
NLL_emofs   = nll_calculate(inputArray, pdf_index_emofs, Z_emofs);
NLL_igmm    = nll_calculate(inputArray, pdf_index_igmm, Z_igmm);
NLL_kde     = nll_calculate(inputArray, pdf_index_kde, Z_kde);
NLL_maf     = nll_calculate(inputArray, pdf_index_maf, Z_maf);
NLL_nsf     = nll_calculate(inputArray, pdf_index_nsf, Z_nsf);
NLL_realnvp = nll_calculate(inputArray, pdf_index_realnvp, Z_realnvp);
NLL_wavelet = nll_calculate(inputArray, pdf_index_wavelet, Z_wavelet);

fprintf("\n=========== FINAL NLL RESULTS ===========\n");
fprintf("EMoFS   : %.6f\n", NLL_emofs);
fprintf("IGMM    : %.6f\n", NLL_igmm);
fprintf("KDE     : %.6f\n", NLL_kde);
fprintf("MAF     : %.6f\n", NLL_maf);
fprintf("NSF     : %.6f\n", NLL_nsf);
fprintf("RealNVP : %.6f\n", NLL_realnvp);
fprintf("Wavelet : %.6f\n", NLL_wavelet);
fprintf("=========================================\n");