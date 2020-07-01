clear;close all;
%% settings
folder = 'Train/DIV2K_train_HR';
savepath = 'Train/DIV2K_4x.h5';
size_input = 25;
size_label = 100;
stride = 100;

%% initialization
data = zeros(1, size_input, size_input, 1);
label = zeros(1, size_label, size_label, 1);
count = 0;

%% generate data
filepaths = [];
filepaths = [filepaths; dir(fullfile(folder, '*.png'))];
filepaths = [filepaths; dir(fullfile(folder, '*.jpg'))];
    
for i = 1 : length(filepaths)
    image = imread(fullfile(folder,filepaths(i).name));
        
    if size(image, 3) == 3
        image = rgb2ycbcr(image);
        image = im2double(image(:, :, 1));

        [hei,wid] = size(image);

        for scale = 4 : 4
            im_label = image(1:hei - mod(hei, stride), 1:wid - mod(wid, stride));
            
            [cropped_hei, cropped_wid] = size(im_label);

            for x = 1 : stride : cropped_hei-size_label+1
                for y = 1 :stride : cropped_wid-size_label+1

                    subim_label = im_label(x : x+size_label-1, y : y+size_label-1);
                    subim_input = imresize(subim_label, 1/scale, 'bicubic');
                    
                    count=count+1;
                    data(1, :, :, count) = subim_input;
                    label(1, :, :,count) = subim_label;
                    
                    % for degree = 0 : 90 : 270

                      %  count=count+1;
                      %  data(1, :, :, count) = imrotate(subim_input, degree);
                      %  label(1, :, :,count) = imrotate(subim_label, degree);

                      %  count=count+1;
                      %  data(1, :, :, count) = fliplr(imrotate(subim_input, degree));
                      %  label(1, :, :,count) = fliplr(imrotate(subim_label, degree));
                    % end
                end
            end
        end
    end
end

order = randperm(count);
data = data(1,:,:,order);
label = label(1,:,:,order); 

%% my writing to HDF5
h5create(savepath, '/data', [1 size_input size_input count]); % width, height, channels, number 
h5create(savepath, '/label', [1 size_label size_label count]); % width, height, channels, number 
h5write(savepath, '/data', data);
h5write(savepath, '/label', label);
h5disp(savepath);
