impath = 'C:\Users\Lenovo\PycharmProjects\avantgarde\data\facebook_frames'
test_files = fileread('filenames.txt')

x = splitlines(test_files);
imdsTrain = imageDatastore(x, 'FileExtensions','.jpg','IncludeSubfolders',1);

%% Create datastore for regression
sigma = 1; %std(double(trainLabels));
mu = 14.9510;

outputSize = [224, 224];
l2Ds = augmentedImageDatastore(outputSize, imdsTrain, 'ColorPreprocessing', 'gray2rgb');

load ('net.mat')

%% Prediction
ytHat = predict(net, l2Ds);
YPredicted = ytHat * sigma + mu;

dold_frames_results = table(YPredicted, imdsTrain.Files);
writetable(dold_frames_results)
