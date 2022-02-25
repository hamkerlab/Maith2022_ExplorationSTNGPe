%Psychtoolbox initialisieren
screennr=0;
KbName('UnifyKeyNames');
[winP, winrect] = Screen('OpenWindow',screennr);
HideCursor(screennr);
xC = winrect(3)/2;
yC = winrect(4)/2;
c.bgPix=[125 125 125]; % Hintergrund
c.fgPix=[0 0 0];
c.ScrRes    = [1920 1080];
c.ScrDist   = 570; %mm
c.ScrWidth  = 523; %mm
c.ScrHeight = 300; %mm
c.ScrHz     = 120; % Hz

%Radius unsichtbarer Kreis
rBigDeg=10; %in grad sehwinkel
rBigPix=c.ScrDist*tan(rBigDeg/180*pi)*(c.ScrRes(1)/c.ScrWidth); %in Pixel

%Radius kleine Kreise
rSmallDeg=0.25; %in grad sehwinkel
rSmallPix=c.ScrDist*tan(rSmallDeg/180*pi)*(c.ScrRes(1)/c.ScrWidth); %in Pixel

%Kreis positions
for idx = [1:12]
    phi=((idx-1)/12)*2*pi-pi/2;
    x=xC+rBigPix*cos(phi);
    y=yC+rBigPix*sin(phi);
    dotPos(:,idx)=[x-rSmallPix,y-rSmallPix, x+rSmallPix,y+rSmallPix];
    dotCenterX(idx) = x;
    dotCenterY(idx) = y;
end







%Begruessung bis Tastendruck
block=1;
begruessung=sprintf('Vielen Dank für die Teilnahme an diesem Experiment. \n\nAllgemeiner Ablauf:\nDies ist Block %d von insgesamt 6 Blöcken. Jeder Block wird nicht länger als 7 Minuten dauern. \nZwischen den Blöcken wird der Versuchsleiter die Tür öffnen und Sie haben die Möglichkeit eine \nkurze Pause zu  machen. \n\nIhre Aufgabe: \nIhre Aufgabe in diesem Experiment wird in allen Blöcken die gleiche sein. Jeder Block besteht aus \nvielen kurzen Durchgängen, deren Ablauf sich durchgehend wiederholt. Zu Beginn eines jeden \nDurchgangs wird ein Fixationskreuz in der Mitte des Bildschirmes gezeigt. Dieses fixieren Sie bitte. \nDas Fixationskreuz verschwindet nach einer kurzen Zeit. Im gleichen Moment tauchen 5 weiße Kreise um \ndie Mitte des Bildschirmes auf. Ihre Aufgabe ist es, so schnell wie möglich einen der Kreise \nanzuschauen. Hierbei können folgende 3 verschiedene Fälle eintreten: \n1. Der Kreis, den Sie anschauen, wird kurz grün. Das bedeutet, dies ist der eine momentan belohnte Kreis. \n2. Der Kreis, den Sie anschauen, wird kurz rot. Das bedeutet, dies ist einer der momentan nicht belohnten Kreise. \n3. Alle Kreise werden kurz blau. Das bedeutet, dass Sie zu langsam waren. Es ist ganz normal, dass \ndieser Fall des Öfteren eintreten wird, da Sie nur sehr wenig Zeit haben einen der 5 Kreise \nanzuschauen. Bitte versuchen Sie dennoch, immer so schnell wie möglich zu reagieren. Nachdem Sie \neines dieser 3 möglichen Feedbacks erhalten haben, verschwinden die 5 Kreise und es erscheint \nwieder das Fixationskreuz in der Mitte. Der Ablauf beginnt von vorn. Ihr Ziel ist es, so oft wie \nmöglich den belohnten Kreis anzuschauen, d.h. das grüne Feedback zu erhalten. Es wird immer nur \nein Kreis gleichzeitig belohnt. Welcher der belohnte Kreis ist, ändert sich nach einigen Durchgängen \n zufällig. Sie werden also in regelmäßigen Abständen erneut nach dem neuen belohnten Kreis suchen \nmüssen. \n\nKalibrierung: \nBevor das Experiment an sich startet, wird zunächst eine Kalibrierung des Systems ablaufen. Hierbei \nwerden Sie nacheinander Kreise auf dem Bildschirm sehen. Schauen Sie diese Kreise an bis Sie nach \neiner kurzen Zeit verschwinden. Solch eine Kalibrierung wird auch während des Experimentes in \nunregelmäßigen Abständen erneut stattfinden, nachdem das Fixationskreuz in der Mitte des Bildschirmes \nzu sehen war. Nach der Kalibrierung startet das Experiment automatisch mit dem Erscheinen des \nFixationskreuzes. \n\nBeliebige Taste drücken, um die Kalibrierung zu starten.',block);
Screen(winP,'FillRect',c.bgPix);
DrawFormattedText(winP,begruessung,'center','center');
Screen(winP,'Flip');
while ~KbCheck, end 
WaitSecs(1);


special1=[1,4,5,8,9,12];
special2=[2,7,10];
special=special;
col=repmat(c.fgPix,[12,1])';
for idx = special1
  col(:,idx)=[100, 100, 100];
end
Screen(winP,'FillRect',c.bgPix);
Screen('FrameOval', winP ,col, dotPos,4);
vbl = Screen(winP,'Flip');
while ~KbCheck, end 
imageArray=Screen('GetImage', winP);
imwrite(imageArray,'bild1.png')
WaitSecs(1);

for idx = special2
  col(:,idx)=[0, 255, 0];
end
Screen(winP,'FillRect',c.bgPix);
Screen('FrameOval', winP ,col, dotPos,4);
vbl = Screen(winP,'Flip');
while ~KbCheck, end 
imageArray=Screen('GetImage', winP);
imwrite(imageArray,'bild2.png')
WaitSecs(1);



  
Screen('CloseAll')
ShowCursor;