% Physionet 데이터 로드 (파일이 없으면 ReadPhysionetData 실행)
if ~isfile('PhysionetData.mat')
    ReadPhysionetData         
end
load PhysionetData

% 신호 데이터와 라벨의 일부 출력
Signals(1:5)'
Labels(1:5)'

% 라벨 요약 출력
summary(Labels)

% 신호 길이 분포를 히스토그램으로 시각화
L = cellfun(@length,Signals);
h = histogram(L);
xticks(0:3000:18000);
xticklabels(0:3000:18000);
title('Signal Lengths')
xlabel('Length')
ylabel('Count')

% 정상 심장 리듬과 심방세동 신호 선택
normal = Signals{1};
aFib = Signals{4};

tiledlayout("flow")
nexttile
plot(normal)
title("Normal Rhythm")
xlim([4000,5200])
ylabel("Amplitude (mV)")
text(4330,150,"P",HorizontalAlignment="center")
text(4370,850,"QRS",HorizontalAlignment="center")

nexttile
plot(aFib)
title("Atrial Fibrillation")
xlim([4000,5200])
xlabel("Samples")
ylabel("Amplitude (mV)")

% 신호 세그먼트화
temp_Signals,Labels] = helperSegmentSignals(Signals,Labels);
Signals(1:5)'
summary(Labels)

% 심방세동(A)과 정상(N) 데이터 분리
afibX = Signals(Labels=="A");
afibY = Labels(Labels=="A");
normalX = Signals(Labels=="N");
normalY = Labels(Labels=="N");

% 데이터셋을 학습/검증/테스트 세트로 분할
rng("default")
indA = splitlabels(afibY,[0.8 0.1],"randomized");
indN = splitlabels(normalY,[0.8 0.1],"randomized");

XTrainA = afibX(indA{1});
YTrainA = afibY(indA{1});
XTrainN = normalX(indN{1});
YTrainN = normalY(indN{1});

XValidA = afibX(indA{2});
YValidA = afibY(indA{2});
XValidN = normalX(indN{2});
YValidN = normalY(indN{2});

XTestA = afibX(indA{3});
YTestA = afibY(indA{3});
XTestN = normalX(indN{3});
YTestN = normalY(indN{3});

% 데이터 증강: 심방세동 데이터를 7배 복제하여 학습 세트 균형 맞춤
XTrain = [repmat(XTrainA,7,1); XTrainN];
YTrain = [repmat(YTrainA,7,1); YTrainN];
XValid = [repmat(XValidA,7,1); XValidN];
YValid = [repmat(YValidA,7,1); YValidN];
XTest = [repmat(XTestA,7,1); XTestN];
YTest = [repmat(YTestA,7,1); YTestN];

% 주파수 분석을 통한 특성 추출
fs = 300;
figure
tiledlayout("flow")
nexttile
pspectrum(normal,fs,"spectrogram",TimeResolution=0.5)
title("Normal Signal")

nexttile
pspectrum(aFib,fs,"spectrogram",TimeResolution=0.5)
title("AFib Signal")

% 즉시 주파수 계산
[instFreqA,tA] = instfreq(aFib,fs);
[instFreqN,tN] = instfreq(normal,fs);

% 즉시 주파수 시각화
figure
tiledlayout("flow")
nexttile
plot(tN,instFreqN)
title("Normal Signal")
xlabel("Time (s)")
ylabel("Instantaneous Frequency")

nexttile
plot(tA,instFreqA)
title("AFib Signal")
xlabel("Time (s)")
ylabel("Instantaneous Frequency")

% 스펙트럼 엔트로피 계산
[s,f,tA2] = pspectrum(aFib,fs,"spectrogram");
pentropyA = spectralEntropy(s,f,Scaled=true);
[s,f,tN2] = pspectrum(normal,fs,"spectrogram");
pentropyN = spectralEntropy(s,f,Scaled=true);

% 스펙트럼 엔트로피 시각화
figure
tiledlayout("flow")
nexttile
plot(tN2,pentropyN)
title("Normal Signal")
ylabel("Spectral Entropy")

nexttile
plot(tA2,pentropyA)
title("AFib Signal")
xlabel("Time (s)")
ylabel("Spectral Entropy")

% 데이터 정규화
temp_XV = [XTrain2{:}];
mu = mean(XV,2);
sg = std(XV,[],2);

XTrainSD = cellfun(@(x)(x-mu)./sg,XTrain2,UniformOutput=false);
XValidSD = cellfun(@(x)(x-mu)./sg,XValid2,UniformOutput=false);
XTestSD = cellfun(@(x)(x-mu)./sg,XTest2,UniformOutput=false);

% LSTM 신경망 구성
layers = [ ...
    sequenceInputLayer(2)
    bilstmLayer(50,OutputMode="last")
    fullyConnectedLayer(2)
    softmaxLayer]

% 학습 옵션 설정
options = trainingOptions("adam", ...
    MaxEpochs=150, ...
    MiniBatchSize=200, ...
    GradientThreshold=1, ...
    Shuffle="every-epoch", ...
    InitialLearnRate=1e-3, ...
    plots="training-progress", ...
    Metrics="accuracy", ...
    InputDataFormats="CTB", ...
    ValidationData={XValidSD,YValid}, ...
    OutputNetwork="best-validation", ...
    Verbose=false);

% 모델 학습
net2 = trainnet(XTrainSD,YTrain,layers,"crossentropy",options);

% 학습 데이터 예측 및 정확도 계산
scores = minibatchpredict(net2,XTrainSD,InputDataFormats="CTB");
trainPred2 = scores2label(scores,classNames);
LSTMAccuracy = sum(trainPred2 == YTrain)/numel(YTrain)*100

% 학습 데이터 혼동 행렬
figure
confusionchart(YTrain,trainPred2,ColumnSummary="column-normalized",...
              RowSummary="row-normalized",Title="Confusion Chart for LSTM");

% 테스트 데이터 예측 및 정확도 계산
scores = minibatchpredict(net2,XTestSD,InputDataFormats="CTB");
testPred2 = scores2label(scores,classNames);
LSTMAccuracy = sum(testPred2 == YTest)/numel(YTest)*100

% 테스트 데이터 혼동 행렬
figure
confusionchart(YTest,testPred2,ColumnSummary="column-normalized",...
              RowSummary="row-normalized",Title="Confusion Chart for LSTM");
