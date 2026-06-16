clear all 
close all 
addpath('..')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Experiment Settings
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define 2D uniform range
x_min = 10; x_max = 40;
% Define parameters
sampleSize = 500;
hp = x_max+(x_max - x_min)*1
HalfPeriod_L = [ hp ];
Mixture_N = 50;
Trimming_Factor = 0.01; 
Convergence_Factor = 0.0000001;
Convergence_Debug = true;
% view settings
my_xlim = hp; 

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
% Finalize input
inputArray = [x_samples_uniform ];

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
result_emofs = Estimated_PDF;
% Plot the estimated pdf
figure;
plot(x_axis, result_emofs);
title("Base Uniform Probability Distribution Function");
xlabel('x (ph)');
ylabel('y (ph)');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% IGMM Estimation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
options = statset('MaxIter',1000, 'Display','final');
gmdist = fitgmdist(inputArray, Mixture_N,'Options',options); % Mixture_N components
grid_points = [x_axis];
f_igmm = pdf(gmdist, grid_points);
% Plot the estimated pdf
figure;
plot(grid_points, f_igmm);
title("Base Uniform Probability Distribution Function");
xlabel('x (ph)');
ylabel('y (ph)');