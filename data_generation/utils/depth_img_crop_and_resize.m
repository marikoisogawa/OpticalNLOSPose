%% Sub function for image cropping and resizing

function cropped_depth_img_list = depth_img_crop_and_resize(param, depth_img_list)
    for depth_imgID = param.framerange_30fps(1) : param.framerange_30fps(2)
        depth_img_cropped(:, :, depth_imgID) = depth_img_list(1:325, 100:424, depth_imgID); % crop to 325*325
        cropped_depth_img_list(:, :, depth_imgID) = imresize(depth_img_cropped(:, :, depth_imgID), [32 32], 'nearest');
    end
end