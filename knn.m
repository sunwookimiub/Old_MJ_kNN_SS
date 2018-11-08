%% knn separation
 
tes=audioread('tes.wav');
ten=audioread('ten.wav');
tex=tes+ten;
 
trs=audioread('trs.wav');
trn=audioread('trn.wav');
trx=trs+trn;
 
sz=1024;
hp=512;
 
wn=hann(sz, 'periodic').^.5;
 
%trS=abs(stft2(trs', sz, hp, 0, wn));
%trN=abs(stft2(trn', sz, hp, 0, wn));
%trX=abs(stft2(trx', sz, hp, 0, wn));
%teX=stft2(tex', sz, hp, 0, wn);
[trS, f, t]=stft(trs', wn, hp, sz, 16000);
[trN, f, t]=stft(trn', wn, hp, sz, 16000);
[trX, f, t]=stft(trx', wn, hp, sz, 16000);
[teX, f, t]=stft(tex', wn, hp, sz, 16000);
trS = abs(trS)
trN = abs(trN)
trX = abs(trX)
 
IBM = trS > trN;
 
D=pdist2(abs(teX'), trX', 'cosine');
[Ds,DsIdx]=sort(D,2,'ascend');
 
K=5;
 
kNNIdx=DsIdx(:,1:K);
kNNIdx1D=reshape(kNNIdx', [numel(kNNIdx),1]);
 
IBM3D=IBM(:,kNNIdx1D);
IBM3D=reshape(IBM3D, size(teX,1), K, []);
IBMEstMed=reshape(median(IBM3D, 2), size(teX,1), []);
IBMEstMean=reshape(mean(IBM3D, 2), size(teX,1), []);
 
%tesReconMed=stft2(teX.*IBMEstMed, sz, hp, 0, wn);
%tesReconMean=stft2(teX.*IBMEstMean, sz, hp, 0, wn);
[tesReconMed, f, t]=stft(teX.*IBMEstMed, wn, hp, sz, 16000);
[tesReconMean, f, t]=stft(teX.*IBMEstMean, wn, hp, sz, 16000);
 
% soundsc(tesReconMed, 16000)
% soundsc(tesReconMean, 16000)
 
minL=min([numel(tesReconMed), numel(tes)]);
 
10*log10( sum(tes(1:minL).^2) / sum((tes(1:minL)-tesReconMed(1:minL)').^2) + eps)
10*log10( sum(tes(1:minL).^2) / sum((tes(1:minL)-tesReconMean(1:minL)').^2) + eps)
