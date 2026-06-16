clear all 
close all 
addpath('..')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Experiment Settings
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define 2D uniform range
x_min = 1; x_max = 10;
% Define parameters
sampleSize = 500;
hp = x_max+(x_max - x_min)*1;
HalfPeriod_L = [ hp hp hp ];
Mixture_N = 25;
Trimming_Factor = 30; 
Convergence_Factor = 0.0000001;
Convergence_Debug = true;
% view settings
my_xlim = hp; 
my_ylim = hp; 
my_zlim = hp;
my_view = [324 15];

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
% Generate x and y samples
rng('shuffle');
x_samples_uniform = x_min + (x_max - x_min) * rand( sampleSize, 1 );
y_samples_uniform = x_min + (x_max - x_min) * rand( sampleSize, 1 );
z_samples_uniform = x_min + (x_max - x_min) * rand( sampleSize, 1 );
% Finalize input
inputArray = [x_samples_uniform y_samples_uniform z_samples_uniform ];
% plot output
figure;
scatter3(x_samples_uniform, y_samples_uniform, z_samples_uniform, ...
        'MarkerEdgeColor','k',...
        'MarkerFaceColor',[0 .75 .75])

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
z_axis_ = pdf_index(:,3);
result = Estimated_PDF;
% Reshape into grid form
x_axis_revised = x_axis(1:MaxPeriod*MaxPeriod);
y_axis_revised = y_axis(1:MaxPeriod*MaxPeriod);
z_axis_revised = z_axis(1:MaxPeriod*MaxPeriod);
% Reshape to coordinate system
X = reshape(x_axis_revised, MaxPeriod, MaxPeriod);
Y = reshape(y_axis_revised, MaxPeriod, MaxPeriod);
Z_emofs = reshape(result, MaxPeriod, MaxPeriod, MaxPeriod);
% Plot the estimated pdf
figure;
surf(X, Y, Z_emofs(:,:,1) ); 
xlabel('x (ph)');
ylabel('y (ph)');
zlabel('Estimated PDF by EMoFS');
set(gca,'ydir','reverse');

% 3D contour
figure;
contour(X, Y, Z_emofs(:,:,1), Mixture_N, 'k');  % 2D contour

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% IGMM Estimation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
options = statset('MaxIter',1000, 'Display','final');
gmdist = fitgmdist(inputArray, Mixture_N,'Options',options); % Mixture_N components
grid_points = [x_axis, y_axis z_axis_];
f = pdf(gmdist, grid_points);
Z_igmm = reshape(f, MaxPeriod, MaxPeriod);
figure;
h1 = surf(X, Y, Z_igmm);
%set(h1, 'FaceAlpha', 0.6);
xlabel('x (ph)');
ylabel('y (ph)');
zlabel('Estimated PDF by IGMM');
set(gca,'ydir','reverse');
%legend('GMM', 'EMoFS');

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