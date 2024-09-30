
clear all
close all
cd 'Add your file extensions'

addpath 'add your file extension'
%Read weather data
Wfilename = ''; % abc = name of data file
Weatherdata = readtable(Wfilename);
DataArray=table2cell(Weatherdata);
% Weather data (X)
% FinalWeatherdata=[num2cell(ones(length(DataArray),1)),DataArray(:,1),DataArray(:,3),DataArray(:,5),DataArray(:,6),DataArray(:,8)];
% FinalWeatherdata=FinalWeatherdata(1:length(FinalLoadData),:);

for i=1:50
filename1='Apt';
filename2=num2str(i);
filename3='_2016.csv';
filename=append(filename1,filename2,filename3);
data=readtable(filename);
data2=table2array(data(:,2));
counter=0;


for j=1:60:502560
if j>length(data2)
break;
end
counter=counter+1;
%Load data (Y)
FinalLoadData(i,counter)=mean(data2(j:j+60));%% final load data for apt fro 350 days

end

end
% Weather data (X)

 FinalWeatherdata1=[DataArray(:,3),...%deleting 5 col
    DataArray(:,16),DataArray(:,17),DataArray(:,19),DataArray(:,13),DataArray(:,15),DataArray(:,18),DataArray(:,20),...
    DataArray(:,11)]; %15,18
%FinalWeatherdata=[num2cell(ones(length(DataArray),1)),DataArray(:,1),...%deleting 5 col
 %   DataArray(:,3),DataArray(:,17),DataArray(:,6),DataArray(:,8),DataArray(:,19)];
FinalWeatherdata2=cell2mat(FinalWeatherdata1);
%FinalWeatherdata=[ones(length(DataArray),1),normalize(FinalWeatherdata2,'range')];
FinalWeatherdata=[ones(length(DataArray),1),FinalWeatherdata2];

FinalWeatherdata=FinalWeatherdata(1:length(FinalLoadData),:);

%X=normalize((cell2mat(FinalWeatherdata)));
X=FinalWeatherdata;
%FinalLoadData=normalize(FinalLoadData','range');

    FinalLoadData=sum(FinalLoadData(1:10,:))
    FinalLoadData=FinalLoadData';

trn = floor(0.7*(length(FinalLoadData)));
trnY = FinalLoadData(1:trn,:);
trnX = FinalWeatherdata(1:trn,:);
%trnX=cell2mat(trnX);

testY = FinalLoadData(trn+1:end,:);
testX = FinalWeatherdata(trn+1:end,:);

net  = fitnet; 
[net,tr] = train(net, trnX', trnY');
perf = perform(net,trnX', trnY')
plotperform(tr);

load_forecast = sim(net,testX')';
%confusion(testY,load_forecast);
err = testY - load_forecast;
errp = (abs(err)./testY)*100;

mae = mean(testY-load_forecast);
mape = mean(errp)
%h1 = figure;
%plotregression(testY',load_forecast)
plot(testY');
hold on
plot(load_forecast,LineStyle="--",Color='r')

for i=1:length(testY)
if(testY(i)<0.5)
    GT(i)=0;
end
if(testY(i)>0.5 & testY(i)<1)
    GT(i)=1;
end
if(testY(i)>1 & testY(i)<1.5)
    GT(i)=2;
end
if(testY(i)>1.5 & testY(i)<2)
    GT(i)=3;
end
if(testY(i)>2 & testY(i)<2.5)
    GT(i)=4;
end
if(testY(i)>2.5 & testY(i)<3)
    GT(i)=5;
end
if(testY(i)>3 & testY(i)<3.5)
    GT(i)=6;
end
if(testY(i)>3.5 & testY(i)<4)
    GT(i)=7;
end
if(testY(i)>4 & testY(i)<4.5)
    GT(i)=8;
end
if(testY(i)>4.5 & testY(i)<5)
    GT(i)=9;
end
if (testY(i)>5)
    GT(i)=10;
end

%for PLoad
if(load_forecast(i)<0.5)
    PL(i)=0;
end
if(load_forecast(i)>0.5 & load_forecast(i)<1)
    PL(i)=1;
end
if(load_forecast(i)>1 & load_forecast(i)<1.5)
    PL(i)=2;
end
if(load_forecast(i)>1.5 & load_forecast(i)<2)
    PL(i)=3;
end
if(load_forecast(i)>2 & load_forecast(i)<2.5)
    PL(i)=4;
end
if(load_forecast(i)>2.5 & load_forecast(i)<3)
    PL(i)=5;
end
if(load_forecast(i)>3 & load_forecast(i)<3.5)
    PL(i)=6;
end
if(load_forecast(i)>3.5 & load_forecast(i)<4)
    PL(i)=7;
end
if(load_forecast(i)>4 & load_forecast(i)<4.5)
    PL(i)=8;
end
if(load_forecast(i)>4.5 & load_forecast(i)<5)
    PL(i)=9;
end
if (load_forecast(i)>5)
    PL(i)=10;
end



end
confusionMatrix = confusionmat(GT,PL);
%onfusionMatrix = confusionmat(testY,load_forecast);
accuracy = sum(diag(confusionMatrix))/sum(sum(confusionMatrix))
