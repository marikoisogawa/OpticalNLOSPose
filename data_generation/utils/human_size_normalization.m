%% Sub function for human size normalization

function normalized_img = human_size_normalization(param, src_img_list)

    normalized_img = zeros(size(src_img_list));
    kinect_size = size( src_img_list(:, :, param.framerange_30fps(1)) );
    human_range = [100 256];
    target_d = 220; % lower is closer
    
    %% initialization
    depth_img_tmp = double(src_img_list);
    depth_img_rotated_tmp = ones(kinect_size(1), kinect_size(2), param.framerange_30fps(2));
    depth_img_scaled_tmp = ones(kinect_size(1), kinect_size(2), param.framerange_30fps(2));
    R = [[1, 0, 0];...
        [0, cosd(param.x_rotation), -sind(param.x_rotation)];...
        [0, sind(param.x_rotation),  cosd(param.x_rotation)]];

    for imageID = param.framerange_30fps(1) : param.framerange_30fps(2)
        %% for human region segmentation
        for y = 1 : 325
            for x = 100 : 424
                depth = depth_img_tmp(y, x, imageID);
                if depth < human_range(1)
                    depth_img_tmp(y, x, imageID) = 1;
                elseif depth > human_range(2)
                    depth_img_tmp(y, x, imageID) = 1;
                end
            end
        end

        %% rotation
        for y = 1 : 325
            for x = 100 : 424
                d = depth_img_tmp(y, x, imageID);
                if d ~= 1
                    % current 3D position
                    X = x;
                    Y = y;
                    Z = d;
                    current3DPos = [X; Y; Z];
                    
                    % rotated 3D position
                    scaled3DPos = R * current3DPos;
                    scaled3DPos(2) = scaled3DPos(2) + 150; % required to avoid head disappear
                    scaled3DPos = round(scaled3DPos);

                    % cropping
                    if scaled3DPos(1) < 1
                        scaled3DPos(1) = 1;
                    elseif scaled3DPos(1) > kinect_size(2)
                        scaled3DPos(1) = kinect_size(2);
                    end

                    if scaled3DPos(2) < 1
                        scaled3DPos(2) = 1;
                    elseif scaled3DPos(2) > kinect_size(1)
                        scaled3DPos(2) = kinect_size(1);
                    end

                    depth_img_rotated_tmp(scaled3DPos(2), scaled3DPos(1), imageID) = current3DPos(3);
                end
            end
        end
        
        tmp = depth_img_rotated_tmp(:, :, imageID);
        human_region = find(tmp > 1);
        diff = median(tmp(human_region)) - target_d;
        
        %% scaling
        for y = 1 : 325
            for x = 50 : 374
                d = depth_img_rotated_tmp(y, x, imageID);
                if d ~= 1
                    % current 3D position
                    X = x;
                    Y = y;
                    Z = d;
                    current3DPos = [X; Y; Z];

                    % Scale
                    scale_param = (256.0 + diff*2.5) / 256.0; % z+z*param = target_z
                    if X <= kinect_size(2) / 2.0 
                        tmp_X = current3DPos(1) - (kinect_size(2)/2.0 - current3DPos(1)) * (scale_param - 0.1);
                    else
                        tmp_X = current3DPos(1) + (current3DPos(1) - kinect_size(2)/2.0) * (scale_param - 0.1);
                    end
                    if Y <= kinect_size(1) / 2.0
                        tmp_Y = current3DPos(2) - (kinect_size(1)/2.0 - current3DPos(2)) * (scale_param + 0.1);
                    else
                        tmp_Y = current3DPos(2) + (current3DPos(2) - kinect_size(1)/2.0) * (scale_param + 0.1);
                    end
                    tmp_Z = current3DPos(3) + current3DPos(3) * scale_param;
                    
                    scaled3DPos = [tmp_X; tmp_Y; tmp_Z];
                    if diff < -70
                        scaled3DPos(2) = scaled3DPos(2) - (70 + diff) * 0.1;
                    else
                        scaled3DPos(2) = scaled3DPos(2) + (70 + diff);
                    end
                    scaled3DPos = round(scaled3DPos);
                    
                    % cropping
                    if scaled3DPos(1) < 1
                        scaled3DPos(1) = 1;
                    elseif scaled3DPos(1) > kinect_size(2)
                        scaled3DPos(1) = kinect_size(2);
                    end

                    if scaled3DPos(2) < 1
                        scaled3DPos(2) = 1;
                    elseif scaled3DPos(2) > kinect_size(1)
                        scaled3DPos(2) = kinect_size(1);
                    end

                    if scaled3DPos(3) < 1
                        scaled3DPos(3) = 1;
                    elseif scaled3DPos(3) > 255
                        scaled3DPos(3) = 255;
                    end

                    % pixel projection
                    depth_img_scaled_tmp(scaled3DPos(2), scaled3DPos(1), imageID) = scaled3DPos(3);
                end
            end
        end
        
        se = strel('square', 3);
        dilated = imdilate(depth_img_scaled_tmp(:, :, imageID), se); % normalize after dilation
        eroded  = imerode(dilated, se);
        normalized_img(:, :, imageID) = imtranslate(eroded, [30, 20], 'FillValues', 1);
    end
end