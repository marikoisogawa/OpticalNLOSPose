%% Main code for the data generation called from data_augmentation_batch.m

function data_augmentation(param)

    fprintf('data augmentation %s -----\n', param.cfg_name);
    
    %% Road depth images and store them to a matrix -----------------------
    for imageID = param.framerange_30fps(1) : param.framerange_30fps(2)
        src_image_name = sprintf('%s/%08d.png', param.srcdir_name, imageID);
        tmp = imread(src_image_name);
        if size(tmp, 3) == 3
            src_img_list(:, :, imageID) = rgb2gray(tmp);
        elseif size(tmp, 3) == 1
            src_img_list(:, :, imageID) = tmp;
        end
    end
    
    
    %% Frame range init ---------------------------------------------------
    param.framenum_30fps = param.framerange_30fps(2) - param.framerange_30fps(1) + 1;
    param.framerange_4fps(1) = 1;
    param.framerange_4fps(2) = floor(param.framenum_30fps / 30.0 * 4.0);
   
    param.temporal_upsample_step = 4./30; % Convert from 4 Hz to 30 Hz
    frame_count_30fps_tmp = 0;
    for ii = param.framerange_4fps(1):param.temporal_upsample_step:param.framerange_4fps(2)
        frame_count_30fps_tmp = frame_count_30fps_tmp + 1;
    end
    param.framerange_30fps(2) = param.framerange_30fps(1) + frame_count_30fps_tmp - 1; % adjust end of the frame to avoid frame size error
    
    
    %% (Optional) Normalize human size ------------------------------------
    if param.human_size_normalization == true
        src_img_list = human_size_normalization(param, src_img_list);
    end
    
    
    %% Crop and resize original depth image -------------------------------
    depth_img_list_tmp = depth_img_crop_and_resize(param, src_img_list);
    clearvars src_img_list
    param.img_size = size(depth_img_list_tmp, 1);
    
    
    %% (Optional) Temporally downsample -----------------------------------
    if param.temporal_downsample == true
        depth_img_list = temporal_downsample(param, depth_img_list_tmp);
        param.current_framerange = param.framerange_4fps;
    else
        depth_img_list = depth_img_list_tmp;
        param.current_framerange = param.framerange_30fps;
    end
    
    
    %% (Optional) Shift temoral peak --------------------------------------
    depth_img_list_tmp = depth_img_list;
    depth_img_list_tmp(:, :, :) = 1;
    human_range = [100 256];
    for imageID = param.current_framerange(1) : param.current_framerange(2)
        depth_img = depth_img_list(:, :, imageID);
        human_index = find((depth_img~=0) & (human_range(1)<depth_img) & (depth_img<human_range(2)));
        [row, col] = ind2sub(size(depth_img), human_index);
        [max_value, max_peak_index] = max(depth_img(human_index));
        [min_value, min_peak_index] = min(depth_img(human_index));
        median_value                = median(depth_img(human_index));
        for i = 1 : size(human_index, 1)
            shifted = (depth_img(row(i), col(i)) - median_value) + param.depth_median;
            shifted = min(shifted, 1024);
            shifted = max(shifted, 1);
            depth_img_list_tmp(row(i), col(i), imageID) = shifted;
        end
    end
    depth_img_list = round(depth_img_list_tmp);
    clearvars non_human_index

    
    %% Get transient images
    fprintf('get 3D depth vol and compute transient...\n');
    for imageID = param.current_framerange(1) : param.current_framerange(2)
        depth_vol = depth_to_vol(param, depth_img_list(:, :, imageID));
        [~, transient_tmp(:, :, :, imageID)] = depth_to_transient(param, depth_vol);
    end
    clearvars depth_vol

    
    %% (Optional) Temporally upsample
    %  
    fprintf('temporal upsample...\n');
    param.transient_max = -realmax;
    param.transient_min =  realmax;
    ii = param.framerange_4fps(1);
    
    for imageID = param.framerange_30fps(1) : param.framerange_30fps(2)
        if param.temporal_downsample == true
            idx_4fps = floor(ii);
            display(idx_4fps);
            img_crop_point = round( (ii-floor(ii))*param.img_size*param.img_size + 1 );

            for index = 1 : param.img_size*param.img_size
                [col, row] = ind2sub([param.img_size, param.img_size], index);
                if index < img_crop_point
                    if idx_4fps+1 > param.framerange_4fps(2)
                        transient_30fps(row, col, :) = transient_tmp(row, col, :, idx_4fps); % to avoid index error
                    else
                        transient_30fps(row, col, :) = transient_tmp(row, col, :, idx_4fps+1);
                    end
                else
                    transient_30fps(row, col, :) = transient_tmp(row, col, :, idx_4fps);
                end
            end
            ii = ii + param.temporal_upsample_step;
            transient_img = transient_30fps;
        else
            transient_img = transient_tmp(:, :, :, imageID);
        end

        % Rotate transient image here, before the .npy saved
        transient_img = flip( imrotate(transient_img, 90), 1);

        for i = 1:size(transient_img, 1)
            for j = 1:size(transient_img, 1)
                meas_slice = transient_img(i, j, :);
                meas_slice = squeeze(meas_slice);
                [max_val, max_idx] = max(meas_slice);
                [pks,locs] = findpeaks(meas_slice);
                pks_img(i, j, imageID) = max_idx;
            end
        end

        tmp_max = max(transient_img(:));
        tmp_min = min(transient_img(:));
        if tmp_max > param.transient_max
            param.transient_max = tmp_max;
        end
        if tmp_min < param.transient_min
            param.transient_min = tmp_min;
        end
        dst_transient_image_name_mat = sprintf('%s/%08d.mat', param.transient_dstdir_name, imageID);
        save(dst_transient_image_name_mat, 'transient_img');
    end 

    clearvars transient_img
    clearvars transient_tmp

    
    %% Normalize and convert from .mat to .npy transient image
    %  Need to perform normalization to avoid memory allocation error
    fprintf('saving transient...\n');
    for imageID = param.framerange_30fps(1) : param.framerange_30fps(2)
        margin = 5;
        if imageID - margin <= param.framerange_30fps(1)
            start_ind = param.framerange_30fps(1);
              end_ind = param.framerange_30fps(1)+2*margin;
        elseif imageID + margin >= param.framerange_30fps(2)
            start_ind = param.framerange_30fps(2)-2*margin;
              end_ind = param.framerange_30fps(2);
        else
            start_ind = imageID-margin;
              end_ind = imageID+margin;
        end

        dst_transient_image_name_mat = sprintf('%s/%08d.mat', param.transient_dstdir_name, imageID);
        dst_transient_image_name_npy = sprintf('%s/%08d.npy', param.transient_dstdir_name, imageID);
        command = sprintf('~/anaconda3/bin/python utils/mat_to_numpy.py --mat_name %s --npy_name %s --matrix_size 32 --enable_scale1 --dstdir %s --start_ind %d --end_ind %d', ...
                            dst_transient_image_name_mat, dst_transient_image_name_npy, param.transient_dstdir_name, start_ind, end_ind);
        dos(command);
    end
    
    for imageID = param.framerange_30fps(1) : param.framerange_30fps(2)
        dst_transient_image_name_mat = sprintf('%s/%08d.mat', param.transient_dstdir_name, imageID);
        command = sprintf('rm %s', dst_transient_image_name_mat);
        dos(command);
    end
 
    fprintf('Done %s -----\n');
end


