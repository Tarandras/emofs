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
e_mu_x = 10; 
e_mu_y = 10;
exp_offset = 20;
mu_x = (x_min_g + x_max_g)/2; % Gaussian parameters
mu_y = (y_min_g + y_max_g)/2;
sigma_x = (x_max - x_min) / 2;
sigma_y = (y_max - y_min) / 2;
sampleSize = 1500; % total samples per distribution
HalfPeriod_L = [ x_max+(x_max - x_min)*5 y_max+(y_max - y_min)*5 ];
Mixture_N = 25; % total number of mixture components
Trimming_Factor = 10; 
Convergence_Factor = 0.000001;
Convergence_Debug = true;
% view preferences
my_xlim = 50; 
my_ylim = 50; 
my_zlim = 0.009;
my_view = [420 29];

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
% Exponential samples
x_samples_exponential = exprnd(e_mu_x, sampleSize, 1) + exp_offset;
y_samples_exponential = exprnd(e_mu_y, sampleSize, 1) + exp_offset;
% Input concatenation & shuffle
x_input = [ x_samples_uniform  x_samples_exponential];
y_input = [ y_samples_uniform  y_samples_exponential];
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
% Exponential
Z_orig_exp = exppdf(X - exp_offset, e_mu_x) .* exppdf(Y - exp_offset, e_mu_y);
% Concatenation & Normalization
Z_orig = Z_orig_uniform + Z_orig_exp;
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
gmdist = fitgmdist(inputArray, Mixture_N,'Options',options);
grid_points = [x_axis, y_axis];
f = pdf(gmdist, grid_points);
Z_igmm = reshape(f, MaxPeriod, MaxPeriod);
% Plot
figure;
h1 = surf(X, Y, Z_igmm);
xlabel('x (in)');
ylabel('y (in)');
zlabel('Estimated PDF by IGMM');
set(gca,'ydir','reverse');


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
figure;
h1 = surf(X, Y, Z_maf, 'EdgeColor','none');
xlabel('x (in)');
ylabel('y (in)');
zlabel('Estimated PDF by MAF');
view(3);

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
figure;
h1 = surf(X, Y, Z_nsf, 'EdgeColor','none');
xlabel('x (in)');
ylabel('y (in)');
zlabel('Estimated PDF by NSF');
view(3);

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
figure;
h1 = surf(X, Y, Z_realnvp, 'EdgeColor','none');
xlabel('x (in)');
ylabel('y (in)');
zlabel('Estimated PDF by Real NVP');
view(3);

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
figure;
h1 = surf(X, Y, Z_wavelet, 'EdgeColor','none');
xlabel('x (in)');
ylabel('y (in)');
zlabel('Estimated PDF by Wavelet');
view(3);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% KDE 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[f, xi, bw] = ksdensity(inputArray, grid_points); 
Z_kde = reshape(f, MaxPeriod, MaxPeriod);
figure;
h1 = surf(X, Y, Z_kde, 'EdgeColor','none');
xlabel('x (in)');
ylabel('y (in)');
zlabel('Estimated PDF by KDE');
view(3);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Draw EMoFS and Original Together
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% create the first surface using colors from the winter colormap
fig = figure(); 
tcl = tiledlayout(fig,1,1); 
ax = nexttile(tcl); 
axEMOFS = ax;
surf1Colormap = winter(); 
tc = colormapToTruecolor(surf1Colormap,Z_orig);  
hsurf1 = surf(ax,X,Y,Z_orig,tc,'FaceColor','interp','EdgeAlpha',0.3);
set(hsurf1, 'FaceAlpha', 0.2);
% create second surface using colors from the autumn colormap
hold on
surf2Colormap = spring(); 
tc = colormapToTruecolor(surf2Colormap,Z_emofs);
hsurf2 = surf(ax,X,Y,Z_emofs,tc,'FaceColor','interp','EdgeAlpha',0.3);
axHidden = axes(tcl,'visible','off','HandleVisibility','off');
colormap(ax,surf1Colormap)
colormap(axHidden,surf2Colormap)
cb1 = colorbar(ax);
cb1.Layout.Tile = 'east';
cb1.Label.String = 'Original PDF';
cb2 = colorbar(axHidden);
cb2.Layout.Tile = 'east';
cb2.Label.String = 'Estimation by EMoFS';
xlim([ -my_xlim my_xlim ]);
ylim([ -my_ylim my_ylim ]);
zlim([0 my_zlim]);
view( my_view );
xlabel('x (in)');
ylabel('y (in)');
zlabel('Original PDF vs EMoFS');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Draw IGMM and Original Together
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% create the first surface using colors from the winter colormap
fig = figure(); 
tcl = tiledlayout(fig,1,1); 
ax = nexttile(tcl); 
axIGMM = ax;
surf1Colormap = winter(); 
tc = colormapToTruecolor(surf1Colormap,Z_orig);  
hsurf1 = surf(ax,X,Y,Z_orig,tc,'FaceColor','interp','EdgeAlpha',0.3);
set(hsurf1, 'FaceAlpha', 0.2);
% create second surface using colors from the autumn colormap
hold on
surf2Colormap = spring(); 
tc = colormapToTruecolor(surf2Colormap,Z_igmm);
hsurf2 = surf(ax,X,Y,Z_igmm,tc,'FaceColor','interp','EdgeAlpha',0.3);
axHidden = axes(tcl,'visible','off','HandleVisibility','off');
colormap(ax,surf1Colormap)
colormap(axHidden,surf2Colormap)
cb1 = colorbar(ax);
cb1.Layout.Tile = 'east';
cb1.Label.String = 'Original PDF';
cb2 = colorbar(axHidden);
cb2.Layout.Tile = 'east';
cb2.Label.String = 'Estimation by IGMM';
xlim([ -my_xlim my_xlim ]);
ylim([ -my_ylim my_ylim ]);
zlim([0 my_zlim]);
view( my_view );
xlabel('x (in)');
ylabel('y (in)');
zlabel('Original PDF vs IGMM');

%linkprop([axEMOFS, axIGMM], ...
%    {'CameraPosition','CameraTarget','CameraUpVector','CameraViewAngle'});

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