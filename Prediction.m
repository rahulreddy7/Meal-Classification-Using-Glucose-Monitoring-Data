mealData = importdata('MealNoMealData/mealData.csv');
mealData = flip(mealData,2);
mealArray = fillmissing(mealData,'nearest',2);
mealArray = fillmissing(mealArray,'nearest',1);
mealData = array2table(mealArray);

zeroes = [];
coefVariation = [];
top_ffts = [];
[n,w]=size(mealData);

for h = 1:n
    %Coefficient Variation
    rowMean = mean(mealData{h,1:w});
    rowStdDev = std(mealData{h,1:w});
    coefVariation = [coefVariation; rowStdDev/rowMean];

    %Zero crossing with CGM Velocity
    cgmVelocity = mealData{h,1:w-1} - mealData{h,2:w}; 
    [G,H] = find(cgmVelocity < 0.5);
    zeroes = [zeroes; H(end)];

    %Fast Fourier Transform
    fft_mealData(h,[1:w])= abs(fft(mealArray(h,[1:w])));  
    top_ffts(h,[1:8]) = fft_mealData(h,[2:9]);
end

%Root Mean Squared
RMS = rms(mealArray,2);

%Feature Matrix
FeatureMatrixMeal = [zeroes RMS coefVariation top_ffts];

%Normalizing the feature matrix between range of 0 to 1
FeatureMatrixMeal = normalize(FeatureMatrixMeal, 'range');

%Performing PCA
[coeff, score, latent] = pca(FeatureMatrixMeal);
our_top_eigens = coeff(:,1:5);

%Updated feature matrix for meal data
new_features_meal = FeatureMatrixMeal * our_top_eigens;

[r,c] = size(new_features_meal);
labelMeal = ones(r,1);

plot(new_features_meal);
xlabel('Day'), ylabel('Value');
title('Top 5 PCA Features Per Day For Patient 1 MEAL data');
legend({'Feature 1','Feature 2','Feature 3','Feature 4','Feature 5'},'Location','northeast');

%%%% No meal calculations

Nomeal = importdata('MealNoMealData/Nomeal.csv');
Nomeal = flip(Nomeal.data,2);
NomealArray = fillmissing(Nomeal,'nearest',2);
NomealArray = fillmissing(NomealArray,'nearest',1);
Nomeal = array2table(NomealArray);

zeroes = [];
coefVariation = [];
top_ffts = [];
[n,w]=size(Nomeal);

for h = 1:n
    %Coefficient Variation
    rowMean = mean(Nomeal{h,1:w});
    rowStdDev = std(Nomeal{h,1:w});
    coefVariation = [coefVariation; rowStdDev/rowMean];

    %Zero crossing with CGM Velocity
    cgmVelocity = Nomeal{h,1:w-1} - Nomeal{h,2:w}; 
    [G,H] = find(cgmVelocity < 0.5);
    zeroes = [zeroes; H(end)];

    %Fast Fourier Transform
    fft_Nomeal(h,[1:w])= abs(fft(NomealArray(h,[1:w])));  
    top_ffts(h,[1:8]) = fft_Nomeal(h,[2:9]);
end

%Root Mean Squared
RMS = rms(NomealArray,2);

%Feature Matrix
FeatureMatrixNoMeal = [zeroes RMS coefVariation top_ffts];

%Normalizing the feature matrix between range of 0 to 1
FeatureMatrixNoMeal = normalize(FeatureMatrixNoMeal, 'range');

%Updated feature matrix for no meal data
new_features_no_meal = FeatureMatrixNoMeal * our_top_eigens;

%Labels for no meal data
[r1,c1] =size(new_features_no_meal);
labelNoMeal = zeros(r1,1);

figure()
plot(new_features_no_meal);
xlabel('Day'), ylabel('Value');
title('Top 5 PCA Features Per Day For Patient 1 NO MEAL data');
legend({'Feature 1','Feature 2','Feature 3','Feature 4','Feature 5'},'Location','northeast');

% Create Training Data and Labels Data
TrainingData = [new_features_meal;new_features_no_meal];
LabelsData = [labelMeal;labelNoMeal];

% k fold cross validation
num_Folds=10;
Indices=crossvalind('Kfold',LabelsData,num_Folds);
for i=1:num_Folds
    TestFoldSamples=TrainingData(Indices==i,:);
    TrainFoldSamples=TrainingData(Indices~=i,:);
    TrainFoldLabel=LabelsData(Indices~=i,:);
    TestFoldLabel=LabelsData(Indices==i,:);
    
    % Train & Test linear SVM
    linearSVM = fitcsvm(TrainFoldSamples,TrainFoldLabel);
    predict_linearSVM=predict(linearSVM,TestFoldSamples);
    
    % Train & Test polynomial SVM
    polynomialSVM = fitcsvm(TrainFoldSamples,TrainFoldLabel, 'KernelFunction', 'polynomial');
    predict_polynomialSVM=predict(polynomialSVM,TestFoldSamples);
    
    %Train & Test Gaussian SVM
    gaussianSVM = fitcsvm(TrainFoldSamples,TrainFoldLabel,'KernelFunction','gaussian');
    predict_gaussianSVM=predict(gaussianSVM,TestFoldSamples);
    
    % Train & Test Naive Bayes
    naiveBayes = fitcnb(TrainFoldSamples,TrainFoldLabel);
    predict_naiveBayes=predict(naiveBayes,TestFoldSamples);
    
    % Calculating accuracy measures for linear SVM, polynomial SVM,
    % Gaussian SVM & Naive Bayes
    linearSVM_accuracy(i,1)=sum(grp2idx(predict_linearSVM)==grp2idx(TestFoldLabel))/length(TestFoldLabel);
    polynomialSVM_accuracy(i,1)=sum(grp2idx(predict_polynomialSVM)==grp2idx(TestFoldLabel))/length(TestFoldLabel);
    gaussianSVM_accuracy(i,1)=sum(grp2idx(predict_gaussianSVM)==grp2idx(TestFoldLabel))/length(TestFoldLabel);
    naiveBayes_accuracy(i,1)=sum(grp2idx(predict_naiveBayes)==grp2idx(TestFoldLabel))/length(TestFoldLabel);
    
    % Creating Confusion matrix for linear, gaussian, polynomial SVM and
    % Naive Bayes
    conf_linearSVM=confusionmat(TestFoldLabel,predict_linearSVM);
    conf_gaussianSVM=confusionmat(TestFoldLabel,predict_gaussianSVM);
    conf_polynomialSVM=confusionmat(TestFoldLabel,predict_polynomialSVM);
    conf_naiveBayes=confusionmat(TestFoldLabel,predict_naiveBayes);
    
    % Calculating Precision, Recall & Fscore
    for p=1:size(conf_linearSVM,1)
        recall_linear(p)=conf_linearSVM(p,p)/sum(conf_linearSVM(p,:));
        recall_gaussian(p)=conf_gaussianSVM(p,p)/sum(conf_gaussianSVM(p,:));
        recall_polynomial(p)=conf_polynomialSVM(p,p)/sum(conf_polynomialSVM(p,:));
        recall_naive(p)=conf_naiveBayes(p,p)/sum(conf_naiveBayes(p,:));
        
        precision_linear(p)=conf_linearSVM(p,p)/sum(conf_linearSVM(:,p));
        precision_gaussian(p)=conf_gaussianSVM(p,p)/sum(conf_gaussianSVM(:,p));
        precision_polynomial(p)=conf_polynomialSVM(p,p)/sum(conf_polynomialSVM(:,p));
        precision_naive(p)=conf_naiveBayes(p,p)/sum(conf_naiveBayes(:,p));
    end
    Recall_linear(i,1)=sum(recall_linear)/size(conf_linearSVM,1);
    Recall_gaussian(i,1)=sum(recall_gaussian)/size(conf_gaussianSVM,1);
    Recall_polynomial(i,1)=sum(recall_polynomial)/size(conf_polynomialSVM,1);
    Recall_naive(i,1)=sum(recall_naive)/size(conf_naiveBayes,1);
    
    Precision_linear(i,1)=sum(precision_linear)/size(conf_linearSVM,1);
    Precision_gaussian(i,1)=sum(precision_gaussian)/size(conf_gaussianSVM,1);
    Precision_polynomial(i,1)=sum(precision_polynomial)/size(conf_polynomialSVM,1);
    Precision_naive(i,1)=sum(precision_naive)/size(conf_naiveBayes,1);
    
    Fscore_linear(i,1)=2*Recall_linear(i,1)*Precision_linear(i,1)/(Recall_linear(i,1)+Precision_linear(i,1));
    Fscore_gaussian(i,1)=2*Recall_gaussian(i,1)*Precision_gaussian(i,1)/(Recall_gaussian(i,1)+Precision_gaussian(i,1));
    Fscore_polynomial(i,1)=2*Recall_polynomial(i,1)*Precision_polynomial(i,1)/(Recall_polynomial(i,1)+Precision_polynomial(i,1));
    Fscore_naive(i,1)=2*Recall_naive(i,1)*Precision_naive(i,1)/(Recall_naive(i,1)+Precision_naive(i,1));
end
save('polynomialSVM.mat','polynomialSVM','our_top_eigens');
save('linearSVM.mat','linearSVM','our_top_eigens');
save('naiveBayes.mat','naiveBayes','our_top_eigens');
save('gaussianSVM.mat','gaussianSVM','our_top_eigens');

disp("Average Recall for linear SVM: "+ mean(Recall_linear));
disp("Average Recall for gaussian SVM: "+ mean(Recall_gaussian));
disp("Average Recall for polynomial SVM: "+ mean(Recall_polynomial));
disp("Average Recall for naive Bayes: "+ mean(Recall_naive));
disp('--------------------------');
disp("Average Precision for linear SVM: "+ mean(Precision_linear));
disp("Average Precision for gaussian SVM: "+ mean(Precision_gaussian));
disp("Average Precision for polynomial SVM: "+ mean(Precision_polynomial));
disp("Average Precision for naive Bayes: "+ mean(Precision_naive));
disp('--------------------------');
disp("Average Fscore for linear SVM: "+ mean(Fscore_linear));
disp("Average Fscore for gaussian SVM: "+ mean(Fscore_gaussian));
disp("Average Fscore for polynomial SVM: "+ mean(Fscore_polynomial));
disp("Average Fscore for naive Bayes: "+ mean(Fscore_naive));
disp('--------------------------');
disp("Average Accuracy for linear SVM: " +mean(linearSVM_accuracy));
disp("Average Accuracy for gaussian SVM: " +mean(gaussianSVM_accuracy));
disp("Average Accuracy for polynomial SVM: " +mean(polynomialSVM_accuracy));
disp("Average Accuracy for naive Bayes: " +mean(naiveBayes_accuracy));
%testScript("C:\Users\rahul reddy\Downloads\mealData1.csv");