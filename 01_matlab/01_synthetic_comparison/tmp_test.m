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
Trimming_Factor = 0; 
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
% Gaussian samples
x_samples_gaussian = mu_x + sigma_x * randn(sampleSize, 1);
y_samples_gaussian = mu_y + sigma_y * randn(sampleSize, 1);
% Poisson samples
x_samples_poisson = poissrnd(p_lambda_x, sampleSize, 1);
y_samples_poisson = poissrnd(p_lambda_y, sampleSize, 1);
% Exponential samples
x_samples_exponential = exprnd(e_mu_x, sampleSize, 1) + exp_offset;
y_samples_exponential = exprnd(e_mu_y, sampleSize, 1) + exp_offset;
% Input concatenation & shuffle
x_input = [x_samples_gaussian x_samples_uniform x_samples_poisson x_samples_exponential];
y_input = [y_samples_gaussian y_samples_uniform y_samples_poisson y_samples_exponential] ;
inputArray = [x_input(:), y_input(:)];
idx = randperm(size(inputArray,1));
inputArray = inputArray(idx, :);

inputArray = [
    [x_samples_gaussian, y_samples_gaussian];
    [x_samples_uniform, y_samples_uniform ];
    [x_samples_poisson,  y_samples_poisson];
    [x_samples_exponential, y_samples_exponential]
];

inputArrayEmofs = [
    [x_samples_gaussian, y_samples_gaussian];
    [y_samples_uniform, x_samples_uniform ];
    [x_samples_poisson,  y_samples_poisson];
    [x_samples_exponential, y_samples_exponential]
];


inputArray = inputArray(randperm(end), :);

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
pdf_index_emofs = pdf_index;
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
NLL_emofs = nll_calculate(inputArray, pdf_index_emofs, Z_emofs)