function pauseFunction(winP, c)
       

    %Eyelink pausieren, Message Pause

    Screen(winP,'FillRect',c.bgPix);
    DrawFormattedText(winP,'Pause \n 2 x beliebige Taste druecken zum fortfahren','center','center');
    Screen(winP,'Flip');
    Eyelink('Message','Pause');
    Eyelink('StopRecording');
    WaitSecs(0.5);
    ResponsePixx('StartNow' ,1,[1 1 1 1 1],1);
    while (sum(ResponsePixx('GetButtons'))==0) end
    while (sum(ResponsePixx('GetButtons'))>0) end
    ResponsePixx('StopNow' ,1,[0 0 0 0 0],0);

    Screen(winP,'FillRect',c.bgPix);
    DrawFormattedText(winP,'Pause \n 1 x beliebige Taste druecken zum fortfahren','center','center');
    Screen(winP,'Flip');
    WaitSecs(0.5);
    ResponsePixx('StartNow' ,1,[1 1 1 1 1],1);
    while (sum(ResponsePixx('GetButtons'))==0) end
    while (sum(ResponsePixx('GetButtons'))>0) end
    ResponsePixx('StopNow' ,1,[0 0 0 0 0],0);

    WaitSecs(0.5);
    
    %Eyelink wieder starten, Message Pause beendet
    
    %Rekalibrieren
    dotrackersetupSM(c.el,13);
    Eyelink('StartRecording');
end