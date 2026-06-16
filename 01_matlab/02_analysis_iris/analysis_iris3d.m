clear all 
close all 

addpath('..')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Read data file
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data = readtable('iris.txt', ...
                 'Delimiter', ',', ...
                 'ReadVariableNames', false);
% Store each column into separate arrays
sepal_length = data{:,1}*1;
sepal_width  = data{:,2}*1;
petal_length = data{:,3}*1;
petal_width  = data{:,4}*1;
species      = data{:,5};

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Experiment Settings
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% view settings
my_xlim = 60; 
my_ylim = 60; 
my_zlim = 0.011;
my_view = [324 15];
% PDF configuration
x_min = floor( min(sepal_length) ); 
x_max = ceil( max(sepal_length) ); 
y_min = floor( min(sepal_width) ); 
y_max = ceil( max(sepal_width) ); 
z_min = floor( min(petal_length) ); 
z_max = ceil( max(petal_length) ); 
a_min = floor( min(petal_width) ); 
a_max = ceil( max(petal_width) ); 
halfPeriod_L_sl = x_max+(x_max - x_min)*1;
halfPeriod_L_sw = y_max+(y_max - y_min)*1;
halfPeriod_L_pl = z_max+(z_max - z_min)*1;
halfPeriod_L_pw = a_max+(a_max - a_min)*1;
Mixture_N = 25;
Trimming_Factor = 5; 
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

mod = py.importlib.import_module('flow_models');
py.importlib.reload(mod);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sample generation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Finalize input
HalfPeriod_L = [ halfPeriod_L_sl halfPeriod_L_pl ...
    halfPeriod_L_sw ] 
MaxHalfPeriod = max(HalfPeriod_L);
MaxPeriod = 2 * MaxHalfPeriod;
[~, rCount] = size(HalfPeriod_L);
memory = MaxPeriod^rCount / 1024 / 1024 / 1024 * 8; 
fprintf("Memory for indexing: %d GB\n", memory);
inputArray = [sepal_length petal_length sepal_width];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EMoFS Estimation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% estimate the pdf with EMoFS
tic
[pdf_index, Estimated_PDF] = emofs( ...
    inputArray, HalfPeriod_L, Mixture_N, Trimming_Factor, ...
    Convergence_Factor, Convergence_Debug); 
toc
% Prepare X,Y,Z indices 
pdf_index_emofs = pdf_index;
MaxPeriod = max(HalfPeriod_L) * 2 + 1;
x_axis = pdf_index(:,1);
y_axis = pdf_index(:,2);
z_axis = pdf_index(:,3);
estimated = Estimated_PDF;
% Reshape into grid form
X = reshape(x_axis, MaxPeriod, MaxPeriod, MaxPeriod);
Y = reshape(y_axis, MaxPeriod, MaxPeriod, MaxPeriod);
Z = reshape(z_axis, MaxPeriod, MaxPeriod, MaxPeriod);
Z_emofs = reshape(estimated, MaxPeriod, MaxPeriod, MaxPeriod);
fprintf("EMoFS finalized ===========\n");
% set grid points for the rest
grid_points = [x_axis, y_axis, z_axis];

NLL_emofs   = nll_calculate(inputArray, pdf_index_emofs, Z_emofs);
fprintf("\n=========== FINAL NLL RESULTS ===========\n");
fprintf("EMoFS   : %.6f\n", NLL_emofs);

fprintf("\n=========== Expecting the Crash for each of the " + ...
    "following methods ===========\n");

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% KDE 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[f, xi, bw] = ksdensity(inputArray, grid_points); 
Z_kde = reshape(f, MaxPeriod, MaxPeriod, MaxPeriod);
% normalize
Z_kde = Z_kde / sum(Z_kde, "all");
% calculate pdf_index for NLL operation
pdf_index_kde = grid_points;
fprintf("KDE finalized ===========\n");

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% IGMM Estimation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
options = statset('MaxIter',1000, 'Display','final');
gmdist = fitgmdist(inputArray, Mixture_N, ...
    'RegularizationValue',1e-5, ...
    'Options',options);
f = pdf(gmdist, grid_points);
Z_igmm = reshape(f, MaxPeriod, MaxPeriod, MaxPeriod);
pdf_index_igmm = grid_points;
fprintf("IGMM finalized ===========\n");

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Masked Autoregressive Flow Estimation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model_type = 'maf';  % 'maf' | 'nsf' | 'realnvp'
result = mod.run_flow( ...
    py.numpy.array(inputArray), ...
    py.numpy.array(grid_points), ...
    model_type);
p = double(result);
Z_maf = reshape(p, MaxPeriod, MaxPeriod, MaxPeriod);
% calculate pdf_index for NLL operation
pdf_index_maf = grid_points;
fprintf("MAF finalized ===========\n");

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Neural Spline Flow Estimation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model_type = 'nsf';  % 'maf' | 'nsf' | 'realnvp'
result = mod.run_flow( ...
    py.numpy.array(inputArray), ...
    py.numpy.array(grid_points), ...
    model_type);
p = double(result);
Z_nsf = reshape(p, MaxPeriod, MaxPeriod, MaxPeriod);
% calculate pdf_index for NLL operation
pdf_index_nsf = grid_points;
fprintf("NSF finalized ===========\n");

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Real-valued Non-Volume Preserving flows
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model_type = 'realnvp';  % 'maf' | 'nsf' | 'realnvp'
result = mod.run_flow( ...
    py.numpy.array(inputArray), ...
    py.numpy.array(grid_points), ...
    model_type);
p = double(result);
Z_realnvp = reshape(p, MaxPeriod, MaxPeriod, MaxPeriod);
% calculate pdf_index for NLL operation
pdf_index_realnvp = grid_points;
fprintf("RealNVP finalized ===========\n");

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
Z_wavelet = reshape(H_smooth, MaxPeriod, MaxPeriod, MaxPeriod);
fprintf("Wavelet finalized ===========\n");


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