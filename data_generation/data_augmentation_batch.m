%% Arguments note:
%  cfg_name: name of the sequence
%  depth_median: temporal peak position (1 ~ 1024)

%% Example command:
%  data_augmentation_batch('0517_take_01', 512);

function data_augmentation_batch(cfg_name, depth_median)


    %% CNLOS reconstruction options ---------------------------------------
    param.alg = 'LCT'; % Reconstruction algorithm 
    param.snr = 8e-1;  % 32e-12;
    param.bin_resolution = 32e-12;
    param.c = 3e8;
    param.z_trim = 0;
    param.width = 1.0;
    param.N = 32;
    param.M = 1024;
    param.range = param.M .* param.c .* param.bin_resolution;
    [param.mtx, param.mtxi] = resamplingOperator(param.M);
    param.psf = definePsf(param.N, param.M, param.width./param.range);
    param.fpsf = fftn(param.psf);

    param.isbackprop = 0; % Toggle backprojection
    param.isdiffuse  = 0; % Toggle diffuse reflection

    if (~param.isbackprop)
        param.invpsf = conj(param.fpsf) ./ (abs(param.fpsf).^2 + 1./param.snr);
    else
        param.invpsf = conj(param.fpsf);
    end

    
    %% Data synthesizing option parameters --------------------------------
    param.x_rotation = 40;
    param.y_translation = 100;
    param.depth_offset = 0;
    param.depth_gain = 1;
    param.depth_median = depth_median;

    param.FWHM = 70e-12; % Full Width Half Max parameter (70ps for Stanford data)
    param.alpha = param.FWHM / 2.35 / param.bin_resolution;

    
    %% Toggle options for 4 types of noises and operations ----------------
    param.poisson_noise = true;
    param.temporal_blur = true;
    param.temporal_downsample = true;
    param.human_size_normalization = true;

    
    %% Name rule ----------------------------------------------------------
    param.cfg_name = cfg_name;
    param.srcdir_name = sprintf('~/datasets/depth/%s', param.cfg_name);
    param.sequence_name = return_sequence_name(param.cfg_name, param);

    
    %% Make dst folder ----------------------------------------------------
    if strcmp(param.cfg_name, 'stanford') == true
        param.dstdir_name = '~/datasets/transient/stanford_32_30fps';
        param.transient_dstdir_name = param.dstdir_name;
    else 
        param.transient_dstdir_name = sprintf('~/datasets/transient/%s', param.sequence_name);
    end
    if ~exist(param.transient_dstdir_name, 'dir'); mkdir(param.transient_dstdir_name); end

    
    %% Frame range --------------------------------------------------------
    if strcmp(param.cfg_name, '0517_take_01') == true
       param.framerange_30fps = [1000, 1200]; % for demo (200 frames only)
       % param.framerange_30fps = [265+169-1,  265+2062+1]; % uncomment this for whole sequence
    elseif strcmp(param.cfg_name, '0517_take_02') == true
        param.framerange_30fps = [132+156-1, 132+2767+1];
    elseif strcmp(param.cfg_name, '0517_take_03') == true
        param.framerange_30fps = [215+53-1, 215+860+1];
    % ---------------------------------------------------------
    elseif strcmp(param.cfg_name, '0702_take_01') == true
        param.framerange_30fps = [240, 4015];
    elseif strcmp(param.cfg_name, '0702_take_02') == true
        param.framerange_30fps = [289, 3883];
    elseif strcmp(param.cfg_name, '0702_take_04') == true
        param.framerange_30fps = [230, 3709];
    elseif strcmp(param.cfg_name, '0702_take_05') == true
        param.framerange_30fps = [207, 3663];
    elseif strcmp(param.cfg_name, '0702_take_06') == true
        param.framerange_30fps = [199, 3844];
    elseif strcmp(param.cfg_name, '0702_take_07') == true
        param.framerange_30fps = [229, 3803];
    elseif strcmp(param.cfg_name, '0702_take_08') == true
        param.framerange_30fps = [261, 3746];
    % ---------------------------------------------------------
    elseif strcmp(param.cfg_name, '1030_take_01') == true
        param.framerange_30fps = [200+0-1, 200+3718+1];
    elseif strcmp(param.cfg_name, '1030_take_02') == true
        param.framerange_30fps = [10+0-1, 10+3693+1];
    elseif strcmp(param.cfg_name, '1030_take_03') == true
        param.framerange_30fps = [-74+76-1, -74+3702+1];
    elseif strcmp(param.cfg_name, '1030_take_04') == true
        param.framerange_30fps = [24+0-1, 24+3828+1];
    elseif strcmp(param.cfg_name, '1030_take_05') == true
        param.framerange_30fps = [6+0-1, 6+3690+1];
    elseif strcmp(param.cfg_name, '1030_take_06') == true
        param.framerange_30fps = [-6+8-1, -6+3738+1];
    % ---------------------------------------------------------
    % Set all range of images
    % ---------------------------------------------------------
    elseif strcmp(param.cfg_name, '1030_take_07') == true
        param.framerange_30fps = [1, 3965];
    elseif strcmp(param.cfg_name, '1030_take_08') == true
        param.framerange_30fps = [1, 2485];
    elseif strcmp(param.cfg_name, '1030_take_09') == true
        param.framerange_30fps = [1, 1068];
    elseif strcmp(param.cfg_name, '1030_take_10') == true
        param.framerange_30fps = [1, 3732];
    elseif strcmp(param.cfg_name, '1030_take_11') == true
        param.framerange_30fps = [1, 3693];
    elseif strcmp(param.cfg_name, '1030_take_12') == true
        param.framerange_30fps = [1, 3976];
    % ---------------------------------------------------------
    elseif strcmp(param.cfg_name, '1030_take_13') == true
        param.framerange_30fps = [1, 3817];
    elseif strcmp(param.cfg_name, '1030_take_14') == true
        param.framerange_30fps = [1, 1963];
    elseif strcmp(param.cfg_name, '1030_take_15') == true
        param.framerange_30fps = [1, 3817];
    elseif strcmp(param.cfg_name, '1030_take_16') == true
        param.framerange_30fps = [1, 3673];
    elseif strcmp(param.cfg_name, '1030_take_17') == true
        param.framerange_30fps = [1, 4064];
    elseif strcmp(param.cfg_name, '1030_take_18') == true
        param.framerange_30fps = [1, 4054];
    % ---------------------------------------------------------
    elseif strcmp(param.cfg_name, '1030_take_19') == true
        param.framerange_30fps = [1, 3890];
    elseif strcmp(param.cfg_name, '1030_take_20') == true
        param.framerange_30fps = [1, 4068];
    elseif strcmp(param.cfg_name, '1030_take_21') == true
        param.framerange_30fps = [1, 3709];
    elseif strcmp(param.cfg_name, '1030_take_22') == true
        param.framerange_30fps = [1, 2778];
    elseif strcmp(param.cfg_name, '1030_take_23') == true
        param.framerange_30fps = [1, 3858];
    elseif strcmp(param.cfg_name, '1030_take_24') == true
        param.framerange_30fps = [1, 3946];
    elseif strcmp(param.cfg_name, 'stanford') == true
        param.framerange_30fps = [0, 0];
    end

    
    %% Main command -------------------------------------------------------
    if strcmp(param.cfg_name, 'stanford') == true
        generate_real_data(param);
    else
        data_augmentation(param);
    end

    
    %% Remove a blank due to Mocap marker drop from 0702_take_01 ----------
    if strcmp(param.cfg_name, '0702_take_01') == true
        % 0702_take_01_1 -----
        tmp_sequence_name_1 = return_sequence_name('0702_take_01_1', param);
        tmp_dstdir_name_1 = sprintf('~/datasets/transient/%s', tmp_sequence_name_1);
        if ~exist(tmp_dstdir_name_1, 'dir'); mkdir(tmp_dstdir_name_1); end
        % 0702_take_01_2 -----
        tmp_sequence_name_2 = return_sequence_name('0702_take_01_2', param);
        tmp_dstdir_name_2 = sprintf('~/datasets/transient/%s', tmp_sequence_name_2);
        if ~exist(tmp_dstdir_name_2, 'dir'); mkdir(tmp_dstdir_name_2); end
        % Copy -----
        for imageID = 240 : 2506
            src_name = sprintf('%s/%08d.npy', param.transient_dstdir_name, imageID);
            dst_name = sprintf('%s/%08d.npy', tmp_dstdir_name_1, imageID);
            command = sprintf('cp %s %s', src_name, dst_name);
            dos(command);
        end
        for imageID = 2757 : 4005
            src_name = sprintf('%s/%08d.npy', param.transient_dstdir_name, imageID);
            dst_name = sprintf('%s/%08d.npy', tmp_dstdir_name_2, imageID);
            command = sprintf('cp %s %s', src_name, dst_name);
            dos(command);
        end

        % Delete -----
        command = sprintf('rm -r %s', param.transient_dstdir_name);
        dos(command);
    end
end


%% Sub-function to return sequence name
function sequence_name = return_sequence_name(cfg_name, param)
    sequence_name = cfg_name;
    if param.poisson_noise == true
        sequence_name = strcat(sequence_name, '_poisson');
    end
    if param.temporal_blur == true
        sequence_name = strcat(sequence_name, '_tmblur');
    end
    if param.temporal_downsample == true
        sequence_name = strcat(sequence_name, '_tmdown');
    end
    if param.human_size_normalization == true
        sequence_name = strcat(sequence_name, '_sizenorm');
    end
    sequence_name = strcat(sequence_name, sprintf('_depth%04d', param.depth_median));
end
