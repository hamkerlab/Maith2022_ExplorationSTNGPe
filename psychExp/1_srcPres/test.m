%
clear;
vp=1;
block=4;

%phase Anz 15 (Exp 1, 2 o. 4) --> 1. TeilBlock 200 Trials; 2. TeilBlock 190 Trials --> 1. TeilBlock 400sec; 2. TeilBlock 380sec
%phase Anz 12 (Exp 3)         --> das Gleiche

%welches Experiment in welchem Block 1-cluster, 2-frequency, 3-blocklength, 4-All
exp=[4,4,3,3,3,3];
%wie viele Phasen/Teilbloecke pro Experiment
phaseAnz=[15,15,12,15];%[15,15,12,15];%[30,30,24,30] 3-blocklength muss weniger Phasen enthalten da diese 30% mehr trials enthalten
%moegliche belohnte Aktionen (abhaengig von VP, 10 verschiedene
%Moeglichkeiten --> 10 VPs) und falls Experiment 2 o. 3 welche davon
%"special" bzw. hauefig
C = nchoosek([1 2 3 4 5],3);
possible_cor=C(vp,:);
special_act=possible_cor(1);

%allgemeine Parameter
exp=exp(block);
phaseAnz=phaseAnz(exp);

% seed random number generators
rand('state',1000*vp+block);
randn('state',1000*vp+block);

%Parameter
if(mod(block,2)==1)
  correctList=makeCorrectList(phaseAnz*2,exp,possible_cor,special_act);
  correctListName=sprintf('../dataRaw/corretList%2.2d%2.2d.mat',vp,block);
  save(correctListName,'correctList')
else
  load(sprintf('../dataRaw/corretList%2.2d%2.2d.mat',vp,block-1));
end

maxcorTrialsAnz=8; %+-1
rewardOnset=0;
tFixMin=1;
tFixMax=4;%in sec
tReactMax=0.4;%in sec
tReward=0.4;%in sec


% filenames
matName = sprintf('BG%2.2d%2.2d_beh.mat',vp,block);
matNameFull = sprintf('../dataRaw/%s',matName);


%Schleife ueber Phasen
if(mod(block,2)==1)
  startPhase=1;
  endPhase=phaseAnz+1;
else
  startPhase=phaseAnz+1;
  endPhase=phaseAnz*2;
end

for phase = [startPhase:endPhase]
    
    %korrekte Aktion festlegen
    correct=correctList(phase);
    fprintf('correct = %d.\n',correct);
    fflush(stdout);
    %max Anzahl an Trials festlegen
    if exp==3 && correct==special_act
        maxcorTrials=randi([2*maxcorTrialsAnz-1,2*maxcorTrialsAnz+1]);
        maxTrials=maxcorTrials+5;
    else
        maxcorTrials=randi([maxcorTrialsAnz-1,maxcorTrialsAnz+1]);
        maxTrials=maxcorTrials+5;
    end
    if (phase==endPhase) && (mod(block,2)==1) %am Ende eines ungeraden Blocks wird die Phase vorzeitig abgebrochen = Pause
        maxTrials=int32(maxcorTrials/2);
    end
    
    %Schleife ueber Trials
    trials=0;
    corTrials=0;
    while ((corTrials<maxcorTrials) && (trials<maxTrials))
      
        WaitSecs(0.5);
        while ~KbCheck, end
        [keyIsDown, secs, keyCode, deltaSecs]=KbCheck;
        taste=KbName(keyCode);
        taste=str2num(taste(1));
        
        
        if taste==correct
          corTrials=corTrials+1;
          fprintf('correct, %d, %d.\n',correct,taste);
          fflush(stdout);
        else
          corTrials=0;
          fprintf('wrong, %d, %d.\n',correct,taste);
          fflush(stdout);
        end
        
        trials=trials+1;
        
    end %trials Schleife
    
end %phase Schleife

% save local variables
save(matNameFull);