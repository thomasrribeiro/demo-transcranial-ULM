% Test coordinate conversion
load('Raw_ultrasonic_data_1s.mat');

% Test pixel in middle of image
z_pix = 100;
x_pix = 120;

R = BFStruct.R_extent(1) + (z_pix - 1) * BFStruct.dR;
Phi = BFStruct.Phi_extent(1) + (x_pix - 1) * BFStruct.dPhi;

fprintf('Pixel (%d, %d) -> R=%.2f, Phi=%.2f deg\n', z_pix, x_pix, R, Phi);

[X, Y] = pol2cart(Phi*pi/180 - pi/2, R);
fprintf('pol2cart: X=%.2f, Y=%.2f\n', X, Y);
fprintf('Z_mm = -Y = %.2f (should be negative for depth)\n', -Y);

% Y from pol2cart with angle-pi/2 will be negative for mid-range R
% So -Y will be positive, which is wrong
% We need Y directly (no negation) since Y is already negative
fprintf('\nCorrection: Z_mm should be Y directly = %.2f\n', Y);
