function test=testScript(fileName)
load('polynomialSVM.mat');
load('linearSVM.mat');
load('naiveBayes.mat');
load('gaussianSVM.mat');
testData=csvread(fileName);
testData = flip(testData,2);
testArray = fillmissing(testData,'nearest',2);
testArray = fillmissing(testArray,'nearest',1);
testData = array2table(testArray);
zeroes = [];
coefVariation = [];
top_ffts = [];
[n,w]=size(testData);
for h = 1:n
    %Coefficient Variation
    rowMean = mean(testData{h,1:w});
    rowStdDev = std(testData{h,1:w});
    coefVariation = [coefVariation; rowStdDev/rowMean];

    %Zero crossing with CGM Velocity
    cgmVelocity = testData{h,1:w-1} - testData{h,2:w}; 
    [G,H] = find(cgmVelocity < 0.5);
    zeroes = [zeroes; H(end)];

    %Fast Fourier Transform
    fft_testData(h,[1:w])= abs(fft(testArray(h,[1:w])));  
    top_ffts(h,[1:8]) = fft_testData(h,[2:9]);
end

%Root Mean Squared
RMS = rms(testArray,2);

%Feature Matrix
testFeatureMatrix = [zeroes RMS coefVariation top_ffts];

%Normalizing the feature matrix between range of 0 to 1
testFeatureMatrix = normalize(testFeatureMatrix, 'range');
updated_feature_matrix = testFeatureMatrix * our_top_eigens;
yp1=predict(polynomialSVM,updated_feature_matrix);
yp2=predict(linearSVM,updated_feature_matrix);
yp3=predict(naiveBayes,updated_feature_matrix);
yp4=predict(gaussianSVM,updated_feature_matrix);
csvwrite('polynomialSVM.csv',yp1);
csvwrite('linearSVM.csv',yp2);
csvwrite('naiveBayes.csv',yp3);
csvwrite('gaussianSVM.csv',yp4);
end
