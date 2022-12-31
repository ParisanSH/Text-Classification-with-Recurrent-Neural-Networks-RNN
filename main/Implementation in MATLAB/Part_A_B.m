%%
% Part A_&_B_Prj_#2
%Farshid Pirbonyeh 40033608

clc
clear
close all

%Importing Datas
Data=readtable('IMDB_Dataset.csv');
Stop_Data=readtable('stopwords.txt');

%Making Labels to be a Number
Catgory = convertvars(Data,{'sentiment'},'categorical');
Data_New = convertvars(Catgory,{'sentiment'},'single');

%Seprating Phrases and Labels
Ph=Data_New{:,1};
La=Data_New{:,2};

%Cell_Data=table2array(Data);
Cell_Stop_Data=(table2array(Stop_Data))';

%Tokenizing
Data_Tok=tokenizedDocument(Ph);
%Stop_Data_Tok=tokenizedDocument(Stop_Data);


%Removing Stop Words
newDocuments = removeWords(Data_Tok,Cell_Stop_Data);
newDocuments1 = joinWords(newDocuments);

%Removing Non_Letter Chars 
for ii=1:5000
    
   splited=split(newDocuments1{ii,1});
   [r1,c1]=size(splited);
   for jj=1:r1
    vra=splited{jj,1};
    pp=double(char(vra));
    o=length(pp);
     r=zeros(1,o);

 for i=1:1:o
    aa=pp(1,i);
  if (64 <aa)&& (aa<91) 
      r(1,i)=aa;
  elseif (96 <aa)&& (aa<123) 
      r(1,i)=aa;
 % elseif aa==32
   %   r(1,i)=aa;
  else
        r(1,i)=1;
  end

 end


m=find(r==1);
rem=sort(m,'descend');
j=length(rem);
for i=1:j
    rr=rem(1,i);
    r(rr)=[];
end


removed=char(r);
splited{jj,1}=removed;
row=string(cellstr(splited'));
joindnew=strjoin(row);

   end
   newDocuments1(ii,1)=joindnew;
end


%Removing Short Words 
newDocuments2= newDocuments1;

for ii=1:5000
    mysr=[];
 b=1; 
   splited1=split(newDocuments1{ii,1});
   [r1,c1]=size(splited1);
   for jj=1:r1
    vra=splited1{jj,1};
    pp=char(vra);
    o=length(pp);
    
  if (o<=2)
      mysr(b,1)=jj;
      b=b+1;
  else
  end
  end
rem=sort(mysr,'descend');
j=length(rem);
for i=1:j
    rr=rem(i,1);
    splited1(rr)=[];
end



row=string(cellstr(splited1'));
joindnew=strjoin(row);
newDocuments2(ii,1)=joindnew;
    
end
 
 %%
 %Final Data that should Change to tokens
Dataa=newDocuments2;
TokD=tokenizedDocument(Dataa);
TokD11 = lower(TokD) ;
%--------
TokD1 = removeWords(TokD11,Cell_Stop_Data);
UNtok = joinWords(TokD1);

bag = bagOfWords(TokD1); 
uniques=unique(bag.Vocabulary,'stable');
%uniques1=unique(enc.Vocabulary,'stable');

      A=UNtok{1,1};
for i=2:1:5000
    B=UNtok{i,1};
    A=cat(2,A,B);
end




%%
figure
wordcloud(TokD1);
title("Training Data")
%%
% Creating Word Seq    
enc = wordEncoding(TokD1);
%sequenceLength = 200;
%sequences = doc2sequence(enc,TokD1,'PaddingDirection','right','Length',sequenceLength);
emb = fastTextWordEmbedding;
sequences = doc2sequence(enc,TokD1,'PaddingDirection','right');
sequenceLength = 200;
sequences1 = doc2sequence(emb,TokD1,'PaddingDirection','right','Length',sequenceLength);
%{
      A1=sequences{1,1};
for i=2:1:5000
    B1=sequences{i,1};
    A1=cat(2,A1,B1);
end

A2=tokenizedDocument(TokD1(1, 1).Vocabulary);  

sequences1 = doc2sequence(emb,A2,'PaddingDirection','right');
sqc=sequences1;
for j=2:1:5000
A2=tokenizedDocument(TokD1(j, 1).Vocabulary);  
sequences1 = doc2sequence(emb,A2,'PaddingDirection','right','Length',852);
sqc=cat(1,sqc,sequences1);
end
%}
%%
%{

%One Hot Encoding Table
seq=[];
for i=1:1:5000
    %oo=categorical(sequences{i,:});
    oo=sequences{i,:};
    seq(i,:)=oo;
end
seq=(seq)+1;
myseq=categorical(seq);
ourseq=seq(:);
uniq=unique(ourseq);

hotmat=zeros(numel(uniq),numel(uniq));
j=1;
for i=1:1:numel(uniq)
    hotmat(i,j)=(hotmat(i,j))+1;
    j=j+1;
end
%}

La=La-1;
[rowss,colss]=size(sequences1);

% Deviding to 80% Train and 20% Test

Labelsi=zeros(4000,1);
Train=(sequences1(1, 1))';


for i=2:1:4000
      T=(sequences1(i, 1))'; 
      Train=cat(1,Train,T)  ;
      Labelsi(i,1)=La(i,1);
end

Labelst=zeros(1000,1);
Test=(sequences1(4001, 1))'  ;

u=1;
for i=4002:5000
        T=(sequences1(i, 1))';
        Test=cat(1,Test,T)  ;
      %Test(u,1)=sequences1(i, 1)  ;
        Labelst(u,1)=La(i,1);
        u=u+1;
      
end

xTRAIN=con2seq(Train);
xTest=con2seq(Test);



%%
%Building Network 1
net = elmannet(1:200,2); %%% NumDelays=200
view(net)
net.trainParam.epochs=1;

T=zeros(4000,200);
class=(Labelsi(:,1));
Targs=horzcat(T,class);

for epoch=1:1:100;
    
    for i=1:1:4000
         xin=xTRAIN(i,:);
         yin=Targs(i,:);
         yy=con2seq(yin);
         net = train(net,xin,yy);
    end
end
view(net)
    







%%
%{

for i=1:1:4000
    sent=zeros(numel(uniq),width(myseq));
    IN=Train(i,:);
    for j=1:1:numel(IN)
        a=IN(1,j);
        sent(:,j)=hotmat(:,a);
    end
    xTRAIN{i,1}=sent;
end
Trains=con2seq(xTRAIN);
for i=1:1:4000
TTrain{i,:}=Train(i,:);
end
%}
%{
for i=1:numel(xTRAIN)
     yTRAIN(i,1)=Labelsi(i,1);
end
%}

yTRAIN=categorical(Labelsi);
yTest=categorical(Labelst);
%%
%{
inputSize = 1;
embeddingDimension = 100;
numHiddenUnits = 200;
numWords = numel(enc.Vocabulary);
numClasses = 2;

layers = [ 
    sequenceInputLayer(inputSize)
    wordEmbeddingLayer(embeddingDimension,numWords)
    lstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];




%}


%%
%Bulding Network 2
 %wordEmbeddingLayer(embeddingDimension,numWords)
numFeatures = 300;
numHiddenUnits = 200;
%embeddingDimension = 50;
%numWords = enc.NumWords;
numClasses = 2;


layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];
maxEpochs = 100;
miniBatchSize=200;
options = trainingOptions('adam', ...
    'GradientThreshold',2, ...
    'InitialLearnRate',0.05, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle','every-epoch', ...
    'Verbose',0, ...
    'Plots','training-progress');
%net1 = configure(xTRAIN,Labelsi,layers,options);
%view(layers)
net = trainNetwork(Train,yTRAIN,layers,options);

%%
%Testing
for i=1:1:1000
    sent=zeros(numel(uniq),width(myseq));
    IN=Test(i,:);
    for j=1:1:numel(IN)
        a=IN(1,j);
        sent(:,j)=hotmat(:,a);
    end
    xTest{i,1}=sent;
end
%{
for i=1:numel(xTest)
     yTest(i,1)=Labelst(i,1);
end
yTest=categorical(yTest);
%}

miniBatchSize=200;
Yout=classify(net,xTest,...
    'MiniBatchSize',miniBatchSize);

Acc=100.*(sum(Yout==yTest)./numel(yTest));
fprintf('\n\nOur Accurcy with only 200 First Words Will be: %d\n\n', Acc)

