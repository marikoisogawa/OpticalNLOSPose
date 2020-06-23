% -------------------------------------------------------------------------
% Reconstruct transient image from synthesized depth image
% All parameters are defined in data_augmentation_batch.m
% 
% This code highly refers the following implementation: 
% M. OÅfToole, D.B. Lindell, G. Wetzstein, 
% ÅgConfocal Non-Line-of-Sight Imaging Based on the Light-Cone TransformÅh, 
% Nature, 2018.
% -------------------------------------------------------------------------

function [transient_img, downsampled_transient] = depth_to_transient(param, depth_vol)

    % Step 2: Resample depth axis and pad result
    z_resampled_rho = zeros(2.*param.M,2.*param.N,2.*param.N);
    z_resampled_rho(1:end./2,1:end./2,1:end./2)  = reshape(param.mtx*depth_vol(:,:),[param.M param.N param.N]);

    % Step 3: Convolve with spf filter
    tmp = ifftn(fftn(z_resampled_rho).*param.fpsf);
    afterPSF_rho = tmp(1:end./2,1:end./2,1:end./2);

    % Step 4: Resample time axis and clamp results
    t_resampled_rho = reshape(param.mtxi*afterPSF_rho(:,:),[param.M param.N param.N]);
    t_resampled_rho = max(real(t_resampled_rho),0);
    
    % Permute data dimensions
    estimated_tau = permute(t_resampled_rho,[3 2 1]);

    % Add temporal blur
    if param.temporal_blur == true 
        for x = 1 : size(estimated_tau, 1)
            for y = 1 : size(estimated_tau, 1)
                before = squeeze( estimated_tau(x, y, :) );
                after = imgaussfilt(before, param.alpha);
                estimated_tau(x, y, :) = after;
            end
        end
    end

    % Normalization (you can change the multiplied value. This effects the amount of Poisson noise)
    estimated_tau = estimated_tau .* 10^4 .* 20;

	% Add Poisson noise
    if param.poisson_noise == true
        noised_with_poisson = poissrnd(estimated_tau);
        estimated_tau  = noised_with_poisson;
    end

    % Temporal up-sampling
    upsampled_estimated_tau = zeros(param.N, param.N, param.M.*4);
    upsampled_estimated_tau(:, :, 1:4:end) = estimated_tau;
    upsampled_estimated_tau(:, :, 2:4:end) = estimated_tau;
    upsampled_estimated_tau(:, :, 3:4:end) = estimated_tau;
    upsampled_estimated_tau(:, :, 4:4:end) = estimated_tau;

    transient_img = upsampled_estimated_tau;
    downsampled_transient = estimated_tau;
    
end
