%%Matlab Code to extract features from time series data using windowing with stride 

stride = 50;
%Sampling frequency
fs = 50;
%Set window size
N = 200;
formatSpec_imu = '%f %f %f %f %f %f';
sizeA_imu = [6 Inf];
formatSpec_det = '%f';

sessions = ["Session01","Session05","Session06","Session07","Session12"];  
num_folders = length(sessions);

for folder = 1 : num_folders    
wristFile = fopen('training_data/'+ sessions(folder) +'/wristIMU.txt','r');
armFile = fopen('training_data/'+ sessions(folder) +'/armIMU.txt','r');
detectionFile = fopen('training_data/'+ sessions(folder) +'/detection.txt','r');

%Read data from wristIMU and armIMU files
detections = fscanf(detectionFile,formatSpec_det); 
wristData = fscanf(wristFile,formatSpec_imu,sizeA_imu);
armData = fscanf(armFile,formatSpec_imu,sizeA_imu);

%Pad N-1 zeros at the beginning
wristData_padded = [zeros(6,N-1) wristData];
armData_padded = [zeros(6,N-1) armData];
tot_length = length(wristData);
feature_length = round(((tot_length-N)/stride)+1) - 1;
features = zeros(feature_length,84);
detections_new = zeros(feature_length,1);

for i = 0:feature_length -1
    % Extract features for each window for wrist data
    k = i*stride + 1;
    a_wrist = wristData(1:3,k:k +N-1);
    g_wrist = wristData(4:6,k:k +N-1);
    a_wrist_feature = m_getFeaturesIMU(a_wrist', fs);
    g_wrist_feature = m_getFeaturesIMU(g_wrist', fs);
    % Extract features for each window for arm data
    a_arm = armData(1:3,k:k +N-1);
    g_arm = armData(4:6,k:k +N-1);
    a_arm_feature = m_getFeaturesIMU(a_arm', fs);
    g_arm_feature = m_getFeaturesIMU(g_arm', fs);
    % Merge both features
    features(i+1,:) = [a_wrist_feature g_wrist_feature a_arm_feature g_arm_feature];
    detections_new(i+1) = detections(k+N-1);
    
end
%Write the whole features into csv file
csvwrite('training_data/'+ sessions(folder)+'_features.csv',features);
csvwrite('training_data/'+ sessions(folder)+'_detections.csv',detections_new);
end
