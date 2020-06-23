% Temporally downsample images from 30fps to 4fps.
% The basic idea is up/down samling of images captured with rolling shutter.
% Please refer our supplementally material pdf for details. 

function [depth_img_4fps] = temporal_downsample(param, depth_img_30fps)
   
    for ii = param.framerange_4fps(1) : param.framerange_4fps(2)
        depth_img_4fps(1:param.img_size, 1:param.img_size, ii) = zeros(param.img_size);
        depth_img_4fps_count(1:param.img_size, 1:param.img_size, ii) = zeros(param.img_size);
    end
    
    % Crop & copy then paste
    idx_30fps = param.framerange_30fps(1);
    depth_img_30fps = double(depth_img_30fps);
    for ii = param.framerange_4fps(1):param.temporal_upsample_step:param.framerange_4fps(2)       
        idx_4fps = floor(ii);
        img_crop_point = round( (ii-floor(ii))*param.img_size*param.img_size + 1 );

        if idx_4fps < param.framerange_4fps(2)
            for index = 1 : param.img_size*param.img_size
                [col, row] = ind2sub([param.img_size, param.img_size], index);
                if index < img_crop_point
                    depth_img_4fps(row, col, idx_4fps+1) = depth_img_4fps(row, col, idx_4fps+1) + depth_img_30fps(row, col, min(param.framerange_30fps(2), idx_30fps+1));
                    depth_img_4fps_count(row, col, idx_4fps+1) = depth_img_4fps_count(row, col, idx_4fps+1) + 1; 
                else    
                    depth_img_4fps(row, col, idx_4fps)  = depth_img_4fps(row, col, idx_4fps) + depth_img_30fps(row, col, idx_30fps);
                    depth_img_4fps_count(row, col, idx_4fps) = depth_img_4fps_count(row, col, idx_4fps) + 1; 
                end
            end

        else
            for index = 1 : param.img_size*param.img_size
                [col, row] = ind2sub([param.img_size, param.img_size], index);
                depth_img_4fps(row, col, idx_4fps)  = depth_img_4fps(row, col, idx_4fps) + depth_img_30fps(row, col, idx_30fps);
                depth_img_4fps_count(row, col, idx_4fps) = depth_img_4fps_count(row, col, idx_4fps) + 1;
            end
        end
        idx_30fps = idx_30fps + 1;
    end
    
    % Normalize
    for ii = param.framerange_4fps(1) : param.framerange_4fps(2)
        depth_img_4fps(:, :, ii) = depth_img_4fps(:, :, ii) ./ depth_img_4fps_count(:, :, ii);
    end
    
end