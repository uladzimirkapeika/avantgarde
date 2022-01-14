clear
addpath('../tools/');
addpath('scripts/');

%% Simulate for 10 iterations
Niter = 1;

fileID = fopen('L2Baseline.txt','w');
for seed = 1:Niter
  params = utkParameters();
  [rmse, mae] = utkL2Baseline(seed, params);
  fprintf(fileID,'%s, %s\n',rmse, mae);
end

fclose(fileID);