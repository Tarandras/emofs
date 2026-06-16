clear all 
close all 
addpath('..')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Read data file
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data = readtable('data.txt', ...
                 'Delimiter', ' ', ...
                 'ReadVariableNames', false);
% Store each column into separate arrays
eruptions = data{:,1}*10; % fine tune resolution
waiting  = data{:,2}*1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Experiment Settings
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% view settings
my_xlim = 60; 
my_ylim = 60; 
my_zlim = 0.011;
my_view = [324 15];
% PDF configuration
x_min = floor( min(eruptions) ); 
x_max = ceil( max(eruptions) ); 
y_min = floor( min(waiting) ); 
y_max = ceil( max(waiting) ); 
halfPeriod_L_e = x_max+(x_max - x_min)*4;
halfPeriod_L_w = y_max+(y_max - y_min)*4;
Mixture_N = 45;
Trimming_Factor = 20; 
Convergence_Factor = 0.000001;
Convergence_Debug = false; 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load Python Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
parentDir = fullfile(pwd, '..');
if count(py.sys.path, parentDir) == 0
    insert(py.sys.path, int32(0), parentDir);
end
py.importlib.invalidate_caches();

try
    mod = py.importlib.import_module('flow_models');
catch ME
    disp(ME.message)
end
py.importlib.reload(mod);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sample generation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Finalize input
HalfPeriod_L = [ halfPeriod_L_e halfPeriod_L_w ] 
MaxHalfPeriod = max(HalfPeriod_L);
MaxPeriod = 2 * MaxHalfPeriod;
[~, rCount] = size(HalfPeriod_L);
memoryGB = MaxPeriod^rCount / 1024 / 1024 / 1024 * 8; 
memoryMB = MaxPeriod^rCount / 1024 / 1024 * 8; 
fprintf("Memory for indexing: %d GB, %d MB\n", memoryGB, memoryMB);
inputArray = [eruptions waiting];
input_X = eruptions;
input_Y = waiting;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EMoFS Estimation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% estimate the pdf with EMoFS
[pdf_index, Estimated_PDF] = emofs( ...
    inputArray, HalfPeriod_L, Mixture_N, Trimming_Factor, ...
    Convergence_Factor, Convergence_Debug); 
% Prepare X,Y,Z indices 
MaxPeriod = max(HalfPeriod_L) * 2 + 1;
pdf_index_emofs = pdf_index;
x_axis = pdf_index(:,1);
y_axis = pdf_index(:,2);
z_axis = Estimated_PDF;
% Reshape into grid form
X = reshape(x_axis, MaxPeriod, MaxPeriod);
Y = reshape(y_axis, MaxPeriod, MaxPeriod);
Z_emofs = reshape(z_axis, MaxPeriod, MaxPeriod);
% Plot the estimated pdf
figure;
surf(X, Y, Z_emofs, 'EdgeColor','none');
xlabel('x (ph)');
ylabel('y (ph)');
zlabel('Estimated PDF by EMoFS');
% 2D contour
figure;
contour(X, Y, Z_emofs, Mixture_N, 'k');  % 2D contour

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% IGMM Estimation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
options = statset('MaxIter',1000, 'Display','final');
gmdist = fitgmdist(inputArray, Mixture_N, ...
    'RegularizationValue', 1e-6, ...
    'Options',options);
grid_points = [x_axis, y_axis];
f = pdf(gmdist, grid_points);
Z_igmm = reshape(f, MaxPeriod, MaxPeriod);
Z_igmm = Z_igmm / sum( Z_igmm, "all" );
pdf_index_igmm = grid_points;
figure;
h1 = surf(X, Y, Z_igmm);
xlabel('x (ph)');
ylabel('y (ph)');
zlabel('Estimated PDF by IGMM');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Masked Autoregressive Flow Estimation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
grid_points = [x_axis, y_axis];
model_type = 'maf';  % 'maf' | 'nsf' | 'realnvp'
result = mod.run_flow( ...
    py.numpy.array(inputArray), ...
    py.numpy.array(grid_points), ...
    model_type);
p = double(result);
z_axis_maf = p; 
Z_maf = reshape(p, MaxPeriod, MaxPeriod);
% normalize
Z_maf = Z_maf / sum(Z_maf, "all");
figure;
h1 = surf(X, Y, Z_maf, 'EdgeColor','none');
xlabel('x (in)');
ylabel('y (in)');
zlabel('Estimated PDF by MAF');
view(3);
% calculate pdf_index for NLL operation
pdf_index_maf = grid_points;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Neural Spline Flow Estimation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model_type = 'nsf';  % 'maf' | 'nsf' | 'realnvp'
result = mod.run_flow( ...
    py.numpy.array(inputArray), ...
    py.numpy.array(grid_points), ...
    model_type);
p = double(result);
z_axis_nsp = p; 
Z_nsf = reshape(p, MaxPeriod, MaxPeriod);
% normalize
Z_nsf = Z_nsf / sum(Z_nsf, "all");
figure;
h1 = surf(X, Y, Z_nsf, 'EdgeColor','none');
xlabel('x (in)');
ylabel('y (in)');
zlabel('Estimated PDF by NSF');
view(3);
% calculate pdf_index for NLL operation
pdf_index_nsf = grid_points;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Real-valued Non-Volume Preserving flows
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model_type = 'realnvp';  % 'maf' | 'nsf' | 'realnvp'
result = mod.run_flow( ...
    py.numpy.array(inputArray), ...
    py.numpy.array(grid_points), ...
    model_type);
p = double(result);
z_axis_realnvp = p;
Z_realnvp = reshape(p, MaxPeriod, MaxPeriod);
% normalize
Z_realnvp = Z_realnvp / sum(Z_realnvp, "all");
figure;
h1 = surf(X, Y, Z_realnvp, 'EdgeColor','none');
xlabel('x (in)');
ylabel('y (in)');
zlabel('Estimated PDF by Real NVP');
view(3);
% calculate pdf_index for NLL operation
pdf_index_realnvp = grid_points;

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
% normalize
Z_wavelet = Z_wavelet / sum(Z_wavelet, "all");
figure;
h1 = surf(X, Y, Z_wavelet, 'EdgeColor','none');
xlabel('x (in)');
ylabel('y (in)');
zlabel('Estimated PDF by Wavelet');
view(3);
% calculate pdf_index for NLL operation
x_centers = (edges1(1:end-1) + edges1(2:end)) / 2;
y_centers = (edges2(1:end-1) + edges2(2:end)) / 2;
[XX, YY] = meshgrid(x_centers, y_centers);
pdf_index_wavelet = [XX(:), YY(:)];
z_axis_wavelet = Z_wavelet(:);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% KDE 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[f_kde, xi, bw] = ksdensity(inputArray, grid_points); 
Z_kde = reshape(f_kde, MaxPeriod, MaxPeriod);
% normalize
Z_kde = Z_kde / sum(Z_kde, "all");
figure;
h1 = surf(X, Y, Z_kde, 'EdgeColor','none');
xlabel('x (in)');
ylabel('y (in)');
zlabel('Estimated PDF by KDE');
view(3);
% calculate pdf_index for NLL operation
pdf_index_kde = grid_points;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NLL EVALUATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

