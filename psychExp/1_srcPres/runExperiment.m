function runExperiment(vp,block)

USE_EYE = 1; % left

% set monitor to backlight mode
Datapixx('Open');
Datapixx('StopAllSchedules');
Datapixx('RegWrRd');
Datapixx('EnableVideoScanningBacklight');
ResponsePixx('Open');
Datapixx('EnableDinDebounce');

%welches Experiment in welchem Block 1-cluster, 2-frequency, 3-blocklength, 4-All
exp=[4,4,1,1,1,1];
%wie viele Phasen/Teilbloecke pro Experiment
phaseAnz=[12,12,9,12];%[30,30,24,30] 3-blocklength muss weniger Phasen enthalten da diese 30% mehr trials enthalten
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
c.bgPix=[125 125 125]; % Hintergrund
c.fgPix=[0 0 0];

%Display Eigneschaften
c.ScrRes    = [1920 1080];
c.ScrDist   = 570; %mm
c.ScrWidth  = 523; %mm
c.ScrHeight = 300; %mm
c.ScrHz     = 120; % Hz

%Radius unsichtbarer Kreis
rBigDeg=10; %in grad sehwinkel
rBigPix=c.ScrDist*tan(rBigDeg/180*pi)*(c.ScrRes(1)/c.ScrWidth); %in Pixel

%Radius innerer unsichtbarer Kreis
rInDeg = 3*rBigDeg/4;
rInPix=c.ScrDist*tan(rInDeg/180*pi)*(c.ScrRes(1)/c.ScrWidth); %in Pixel

%Radius kleine Kreise
rSmallDeg=0.25; %in grad sehwinkel
rSmallPix=c.ScrDist*tan(rSmallDeg/180*pi)*(c.ScrRes(1)/c.ScrWidth); %in Pixel

% toleranz Fixation
tolDeg = 1.5;
tolPix = c.ScrDist*tan(tolDeg/180*pi)*(c.ScrRes(1)/c.ScrWidth); %in Pixel

%Psychtoolbox initialisieren
screennr=1;
KbName('UnifyKeyNames');
[winP, winrect] = Screen('OpenWindow',screennr);
HideCursor(screennr);

xC = winrect(3)/2;
yC = winrect(4)/2;

% filenames
edfName = sprintf('BG%2.2d%2.2d.edf',vp,block);
matName = sprintf('BG%2.2d%2.2d_beh.mat',vp,block);
edfNameFull = sprintf('../dataRaw/%s',edfName);
matNameFull = sprintf('../dataRaw/%s',matName);



if exist(edfNameFull)|exist(matNameFull)
    warning('data file exists, type dbcont to continue');
    keyboard;
end

% initialize eyelink
c.el = initeyelinkdefaultsSM(winP,c.bgPix,c.fgPix);
c.el.calibrationtargetsize = rSmallPix;
c.el.calibrationtargetwidth = 6;
initEL1000SM(edfName,c.ScrRes,c.ScrDist,c.ScrWidth,c.ScrHeight);
Eyelink('command', 'calibration_area_proportion = 0.6, 0.8')
Eyelink('command', 'validation_area_proportion = 0.6, 0.8')



%Kreis positions
for idx = [1 2 3 4 5]
    phi=((idx-1)/5)*2*pi-pi/2;
    x=(winrect(3))/2+rBigPix*cos(phi);
    y=(winrect(4))/2+rBigPix*sin(phi);
    dotPos(:,idx)=[x-rSmallPix,y-rSmallPix, x+rSmallPix,y+rSmallPix];
    dotCenterX(idx) = x;
    dotCenterY(idx) = y;
end

%Begruessung bis Tastendruck
begruessung=sprintf('Vielen Dank fuer die Teilnahme an diesem Experiment. \n\nAllgemeiner Ablauf:\nZwischen den Bloecken wird der Versuchsleiter die Tuer oeffnen und Sie haben die Moeglichkeit eine \nPause zu  machen. \n\nIhre Aufgabe: \nIhre Aufgabe in diesem Experiment wird in allen Bloecken die gleiche sein. Jeder Block besteht aus \nvielen kurzen Durchgaengen, deren Ablauf sich wiederholt. Zu Beginn eines jeden Durchgangs \nwird ein Fixationskreuz in der Mitte des Bildschirmes gezeigt. Dieses fixieren Sie bitte. \nDas Fixationskreuz verschwindet nach einer kurzen Zeit. Im gleichen Moment tauchen 5 Kreise um \ndie Mitte des Bildschirmes auf. Ihre Aufgabe ist es, so schnell wie moeglich einen der Kreise \nanzuschauen. Hierbei koennen folgende 3 verschiedene Faelle eintreten: \n1. Der Kreis, den Sie anschauen, wird kurz gruen. Das bedeutet, dies ist der eine momentan belohnte Kreis. \n2. Der Kreis, den Sie anschauen, wird kurz rot. Das bedeutet, dies ist einer der momentan nicht belohnten Kreise. \n3. Alle Kreise werden kurz blau. Das bedeutet, dass Sie zu langsam waren. Es ist vollkommen ueblich und zu erwarten, dass \ndieser Fall haeufiger eintreten wird, da Sie nur sehr wenig Zeit haben einen der 5 Kreise anzuschauen. \nBitte versuchen Sie dennoch, immer so schnell wie moeglich zu reagieren. Nachdem Sie \neines dieser 3 moeglichen Feedbacks erhalten haben, verschwinden die 5 Kreise und es erscheint \nwieder das Fixationskreuz in der Mitte. Der Ablauf beginnt von vorn. \n\nIhr Ziel: \nFinden Sie immer, so schnell wie moeglich, den belohnten Kreis und versuchen Sie rotes Feedback zu vemeiden! \nEs gibt immer nur einen belohnten Kreis, dieser wird innerhalb eines Blocks mehrmals die Position wechseln. \n\nKalibrierung: \nBevor das Experiment an sich startet, wird zunaechst eine Kalibrierung des Systems ablaufen. Hierbei \nwerden Sie nacheinander Kreise auf dem Bildschirm sehen. Schauen Sie diese Kreise an bis Sie nach \neiner kurzen Zeit verschwinden. Solch eine Kalibrierung wird auch waehrend des Experimentes in \nunregelmaessigen Abstaenden erneut stattfinden, nachdem das Fixationskreuz in der Mitte des Bildschirmes \nzu sehen war. Nach der Kalibrierung startet das Experiment automatisch mit dem Erscheinen des \nFixationskreuzes. \n\nBeliebige Taste druecken, um die Kalibrierung zu starten.',block);
Screen(winP,'FillRect',c.bgPix);
DrawFormattedText(winP,begruessung,'center','center');
Screen(winP,'Flip');
ResponsePixx('StartNow' ,1,[1 1 1 1 1],1);
while (sum(ResponsePixx('GetButtons'))==0) end
while (sum(ResponsePixx('GetButtons'))>0) end
ResponsePixx('StopNow' ,1,[0 0 0 0 0],0);


WaitSecs(1);


%Kalibrieren
dotrackersetupSM(c.el,13);
Eyelink('StartRecording');


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
    fprintf('\n phase = %d; correct = %d.\n',phase,correct);
    %fflush(stdout);
    %max Anzahl an Trials festlegen
    if exp==3 && correct==special_act
        maxcorTrials=randi([2*maxcorTrialsAnz-1,2*maxcorTrialsAnz+1]);
        maxTrials=maxcorTrials+5;
    else
        maxcorTrials=randi([maxcorTrialsAnz-1,maxcorTrialsAnz+1]);
        maxTrials=maxcorTrials+5;
    end
    if (phase==endPhase) && (mod(block,2)==1) %am Ende eines ungeraden Blocks wird die Phase vorzeitig abgebrochen = Pause zwischen Teilbloecken
        maxTrials=int32(maxcorTrials/2);
    end
    
    %Schleife ueber Trials
    trials=0;
    corTrials=0;
    while ((corTrials<maxcorTrials) && (trials<maxTrials))
        
        found = 0;
        while (~found)
            
            %Fixationskreuz tReward nach reward (aller erstes sofort)
            Screen(winP,'FillRect',c.bgPix);
            Screen('DrawLine',winP, [0 0 0], winrect(3)/2-10,(winrect(4))/2, winrect(3)/2+10,(winrect(4))/2,2);
            Screen('DrawLine',winP, [0 0 0], winrect(3)/2,(winrect(4))/2-10, winrect(3)/2,(winrect(4))/2+10,2);
            vbl = Screen(winP,'Flip',rewardOnset+tReward);
            Eyelink('Message',sprintf('fixation on; %15.10f',vbl));
            
            found = WaitUntilFound(c.el,winrect(3)/2,winrect(4)/2,tolPix,tFixMin,tFixMax);
            
            if ~found
                %Screen(winP,'FillRect',c.bgPix);
                %DrawFormattedText(winP,'Rekalibrieren','center','center');
                fprintf('recalibrate.\n');
                Eyelink('Message',sprintf('fixation failed; %15.10f',vbl));
                %Screen(winP,'Flip');
                dotrackersetupSM(c.el,13);
                Eyelink('StartRecording');
            end
            
        end % found==1
        Eyelink('Message',sprintf('fixation done; %15.10f',vbl));
        
        %Stimulus / 5 Objekte
        Screen(winP,'FillRect',c.bgPix);
        Screen('FrameOval', winP ,c.fgPix, dotPos,4);
        vbl = Screen(winP,'Flip');
        
        %EyelinkMessage StimulusOnset + welcher correct
        Eyelink('Message',sprintf('stimulus on; correct: %d; %15.10f',correct,vbl));
        
        %ueberschreitet Blick inneren Kreis innerhalb tReactMax?
        isOut = 0;
        start=GetSecs;
        tCurrent = start;
        while (~isOut)&&((tCurrent-start)<tReactMax)
        
            sample=Eyelink( 'newfloatsampleavailable');
            tCurrent = GetSecs();
            if sample > 0
                evt = Eyelink( 'newestfloatsample');
                xE = evt.gx(USE_EYE);
                yE = evt.gy(USE_EYE);
                pupil=evt.pa(USE_EYE);
                
                if ((xE~=c.el.MISSING_DATA) && (yE~=c.el.MISSING_DATA) && (pupil>0))  % pupil visisible? */
                    % within radius
                    isOut = ((xE-xC)^2+(yE-yC)^2>rInPix^2);
                else
                    isOut =0;
                end
            end
            
        end
        
        col=repmat(c.bgPix,[5,1])';
        if isOut
            % fast response --> which is the closest dot?
            distToDots2 = (dotCenterX-xE).^2+(dotCenterY-yE).^2;
            [~,nearestDotIdx]  = min(distToDots2);
            
            if nearestDotIdx==correct
                Eyelink('Message',sprintf('Dot selected: %d; correct;fast',nearestDotIdx));
                col(:,nearestDotIdx)=[0, 255, 0];
                corTrials=corTrials+1;
                fprintf('correct, %d, %d.\n',correct,nearestDotIdx);
            else
                Eyelink('Message',sprintf('Dot selected: %d; wrong',nearestDotIdx));
                col(:,nearestDotIdx)=[255, 0, 0];
                corTrials=0;
                fprintf('wrong, %d, %d.\n',correct,nearestDotIdx);
            end
        else
            % timeout
            col(:,:)=repmat([0, 0, 255],5,1)';
            Eyelink('Message','Dot timeout');
            fprintf('timeout.\n');
        end
        
        %Reward
        Screen(winP,'FillRect',c.bgPix);
        Screen('FrameOval', winP ,col, dotPos,4);
        rewardOnset=Screen(winP,'Flip');
        Eyelink('Message','response shown');
        
        trials=trials+1;
        
    end %trials Schleife
    
end %phase Schleife
fprintf('Block beendet.\n');
%Block beendet (Abschlusstext anstatt neues Fixationskreuz)
Screen(winP,'FillRect',c.bgPix);
DrawFormattedText(winP,'Block beendet','center','center');
Screen(winP,'Flip',rewardOnset+tReward);


% save local variables
save(matNameFull);

Eyelink('StopRecording');
WaitSecs(1);
Eyelink('CloseFile');
WaitSecs(1);
disp('Eyelink file closed')
Eyelink('ReceiveFile',edfName,edfNameFull);
disp('edf saved')
WaitSecs(1);
Eyelink('Shutdown');
Screen('CloseAll')
ShowCursor;
ResponsePixx('Close');
end
