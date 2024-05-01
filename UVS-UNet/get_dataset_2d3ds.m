%     Copyright (C) 2024 Authors of UVS-CNNs
% 
%     This program is free software: you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published by
%     the Free Software Foundation, either version 3 of the License, or
%     (at your option) any later version.
% 
%     This program is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.
% 
%     You should have received a copy of the GNU General Public License
%     along with this program.  If not, see <https://www.gnu.org/licenses/>.
%%
clc;clear;
load('semantic_labels.mat')
title_area = {'1','2','3','4','5a','5b','6'};
class_name = {'ceiling', 'floor', 'wall', 'column', 'beam', 'window', 'door', 'table', 'chair', 'bookcase', 'sofa', 'board', 'clutter'};
s_label = size(labels,2);
se_lb = uint8(ones(1,s_label)*255);
for ii = 1:s_label
    lb = labels{ii};
    lb = strsplit(lb,'_');
    [ism,id] = ismember(lb(1), class_name);
    if ism
        se_lb(ii) = id - 1;
    end
end
for m = 1:7
    num_area = title_area{m};
    files = dir(['2D3DS\area_',num_area,'\semantic\*.png']);
    for n = 1 : numel(files) % par
        tic;
        name_image = files(n).name;
        A = imread(['2D3DS\area_',num_area,'\semantic\',name_image]);
%         B = imread(['2D3DS\area_',num_area,'\semantic_pretty\',name_image(1:end-4),'_pretty.png']);
%         figure;
%         imshow(B)
        index = double(A(:,:,1)) * 256  * 256 + double(A(:,:,2)) * 256 + double(A(:,:,3));
        index = index + 1;
        index(index(:,:) > 9816) = 1;
        M = se_lb(index);
        imwrite(M,['2D3DS\mask_',num_area,'_',num2str(n),'.png'])
%         figure;
%         imshow(M)
        toc;
    end
end

%%
clc;clear;
title = '13';
title_area = {'1','2','3','4','5a','5b','6'};
load(['simg_data\Vertices_',title,'.mat']);
load(['simg_data\seg_list_2d_',title,'.mat']);

for i = 1:7
    num_area = title_area{i};
    files = dir(['2D3DS\area_',num_area,'\rgb\*.png']);
    parfor n = 1 : numel(files)
        tic;
        name_image = files(n).name;
        name_image = name_image(1:end-7);
        num = size(segmentationList,1);
        num_img = 9;
        rand_num = floor(rand(1,num_img)*(num-1)*0.5) + 1;
        for m = 1 : num_img
            start_index = rand_num(m);
            disp(num2str(start_index))
            A = imread(['2D3DS\area_',num_area,'\rgb\',name_image,'rgb.png']);
            D = imread(['2D3DS\area_',num_area,'\depth\',name_image,'depth.png']);
            M = imread(['2D3DS\mask_2d\mask_',num_area,'_',num2str(n),'.png']);
            A = imresize(A,[num,num]);
            D = imresize(D,[num,num]);
            D = double(D) / 65535;
            M = imresize(M,[num,num],"nearest");
            
            if m == 2
                HSV=rgb2hsv(A);
                V=HSV(:,:,3);
                V=adapthisteq(V);
                HSV(:,:,3)=V;
                A=uint8(hsv2rgb(HSV)*255);
            elseif m > 2
                A = [A(:,start_index:end,:),A(:,1:start_index,:)];
                D = [D(:,start_index:end),D(:,1:start_index)];
                M = [M(:,start_index:end),M(:,1:start_index)];
            end
            [r,~] = size(Vertices);
            Simage = zeros(r,5);
            Simage_mask = zeros(r,1);
            for j = 1:num
                for k = 1:num
                    count = Simage(segmentationList(j,k),5);
                    rgbd_data = [double(A(j,k,1)),double(A(j,k,2)),double(A(j,k,3)),double(D(j,k))];
                    Simage(segmentationList(j,k),1:4) = (Simage(segmentationList(j,k),1:4)*count+rgbd_data)/(count+1);
                    Simage(segmentationList(j,k),5) = count + 1;
                    Simage_mask(segmentationList(j,k),1) = M(j,k);
                end
            end
            depth = Simage(:,4)*20;
            depth(depth(:,:) > 1) = 1;
            SimageForPy = [Simage(:,1:3)/255,depth];
            SimageForPy_mask = uint8(Simage_mask(:,1));
            parsave(['dataset\Simage_',title,'_RGBD\',num_area,'_',num2str(n),'_',num2str(m),'.mat'], SimageForPy);
            parsave(['dataset\Simage_',title,'_mask\',num_area,'_',num2str(n),'_',num2str(m),'.mat'], SimageForPy_mask);
        end
        toc;
    end
    
end


%%
clc;clear;
title = '13';
files = dir(['dataset\Simage_',title,'_RGBD\*.mat']);

%     training   testing
% 1   1,2,3,4,6    5
% 2   1,3,5,6      2,4
% 3   2,4,5        1,3,6

train_area_all  = {{'1','2','3','4','6'},{'1','3','5','6'},{'2','4','5'}};
tic;
for i = 1 : 3
    area_title = num2str(i);
    train_area = train_area_all{i};
    fidID_train = fopen(['dataset_title\',area_title,'\dataset_title_train.txt'],'w+');
    fidID_test  = fopen(['dataset_title\',area_title,'\dataset_title_val.txt'],'w+');
    for n = 1 : numel(files)
        name_image = files(n).name;
        index_area = name_image(1);
        if ismember(name_image(1),train_area)
            fprintf(fidID_train,[name_image,'\n']);
        elseif name_image(end-4) == '1'
            fprintf(fidID_test,[name_image,'\n']);
        end
    end
end
toc;
fclose(fidID_train);
fclose(fidID_test);


function [] = parsave(dir,SimageForPy)
    save(dir,'SimageForPy');
end

