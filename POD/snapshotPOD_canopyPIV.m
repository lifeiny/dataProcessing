% reconstruction
clear
format long
z = 50;
Tr =2;
file_name1 = ['/home/lifei/HDD_TOWER/data_f/lifei//PIV_Canopy/data/A1_H0',num2str(z),'_T',num2str(Tr),'.n8.nc'];
file_name2 = ['/home/lifei/HDD_TOWER/data_f/lifei//PIV_Canopy/data/A2_H0',num2str(z),'_T',num2str(Tr),'.n8.nc'];
file_name3 = ['/home/lifei/HDD_TOWER/data_f/lifei//PIV_Canopy/data/A3_H0',num2str(z),'_T',num2str(Tr),'.n8.nc'];

finfo=ncinfo(file_name1);
x = ncread(file_name1, 'x');
y = ncread(file_name1, 'y');
z = ncread(file_name1, 'z');
T = ncread(file_name1, 'time'); %snapshot
U1=ncread(file_name1, 'u');
U2=ncread(file_name2, 'u');
U3=ncread(file_name3, 'u');
U = [U1,U2,U3];
A1_U_mean = mean(U1(:,:,1,:),4);
A2_U_mean = mean(U2(:,:,1,:),4);
A3_U_mean = mean(U3(:,:,1,:),4);

[A1_x, A1_y] = find(A1_U_mean ~=-999);
[A2_x, A2_y] = find(A2_U_mean ~=-999);
[A3_x, A3_y] = find(A3_U_mean ~=-999);

U_mean = mean(U(:,:,1,:),4); % mean velocity
u_flu = U-U_mean;    % fluctuation velocity

U_1d = U(:,:,1,1); % To 1-D data
U_1d(U_1d==-999)=[]; 
sampling_point = length(U_1d); % exclude the invalid value
clear U_1d;


U_time = zeros(sampling_point,length(T)); %Blank matrix

for t = 1:length(T)
    U_1d = U(:,:,1,t); % To 1-D data
    U_1d(U_1d==-999)=[];  %row sampling position, column time
    U_time(:,t) = U_1d'; %row sampling position, column time

    fprintf("process %d \n",t);
end

u_time = U_time - mean(U_time, 2);



fprintf('processing u');

correlation_u = u_time'*u_time/sampling_point; %correlation matrix
[beta_u,lamda_u]=eig(correlation_u); % get eig and matrix
clear correlation_u;


diag_lamda_u =sort(diag(lamda_u),'descend');
phi = u_time*beta_u;
phi_nor = phi./sqrt(sum(phi.*phi));
timecoefficient_u = u_time'*phi_nor;


AX_reconstruction_time = zeros(50,150,20000);

for time_step = 1:20000
% modenumber = 19950:20000;
mode_n = 4;
modenumber = sort(20000-mode_n+1);
u_mode = timecoefficient_u(:,modenumber)*phi_nor(:,modenumber)';


U1_tem = U1(:,:,1,1);
U1_tem(U1_tem==-999)=[];
U2_tem = U2(:,:,1,1);
U2_tem(U2_tem==-999)=[];
U3_tem = U3(:,:,1,1);
U3_tem(U3_tem==-999)=[];

A1_length = length(U1_tem);
A2_length = length(U2_tem);
A3_length = length(U3_tem);

A1 = u_mode (:,1:A1_length);
A2 = u_mode(:,A1_length+1:A1_length+A2_length);
A3 = u_mode(:,A1_length+A2_length+1:A1_length+A2_length+A3_length);

A1_time_step = A1(time_step,:);
A2_time_step = A2(time_step,:);
A3_time_step = A3(time_step,:);

%process A1
A1_u_reconstruction_time = zeros(50,50);
for A1_i = 1:length(A1_x)
    
    A1_u_reconstruction(A1_x(A1_i),A1_y(A1_i)) = A1_time_step(A1_i);


end

A1_U_reconstruction = A1_u_reconstruction + A1_U_mean;

%process A2
A2_u_reconstruction = zeros(50,50);
for A2_i = 1:length(A2_x)
    A2_u_reconstruction(A2_x(A2_i),A2_y(A2_i)) = A2_time_step(A2_i);
end

A2_U_reconstruction = A2_u_reconstruction + A2_U_mean;

%process A3
A3_u_reconstruction = zeros(50,50);
for A3_i = 1:length(A3_x)
    A3_u_reconstruction(A3_x(A3_i),A3_y(A3_i)) = A3_time_step(A3_i);
end
A3_U_reconstruction = A3_u_reconstruction + A3_U_mean;

AX_reconstruction_time(:,:,time_step) = [A1_U_reconstruction',A2_U_reconstruction',A3_U_reconstruction'];

AX_original = [U1(:,:,1,time_step)',U2(:,:,1,time_step)',U3(:,:,1,time_step)'];


mode_n(end);
time_step
end
h5FileName = '/home/lifei/HDD_TOWER/data_f/lifei/PIV_Canopy/code/POD/reconstructionSingle';
fileName = [h5FileName,'/AX_Resconstruction_H0', num2str(z),'_T',num2str(Tr), '_mode', num2str(mode_n(end)),'single.h5'];

h5create(fileName,'/u',size(AX_reconstruction_time));
h5write(fileName,'/u',AX_reconstruction_time);


% h1=figure(1);
% set(h1,'position',[100,100,1500,500]);
% levels = -2:0.01:2;
% [C, H] = contourf(AX_reconstruction,levels);
% colorbar;
% colormap('jet');
% set(H,'Linecolor', 'none');
% saveas(h1,['AX_H0',num2str(z),'_T',num2str(Tr),'_reconstrut_time_',num2str(time_step),'_mode_',num2str(mode_n(end))],'png')