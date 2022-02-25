function [found] = WaitUntilFound(el,x,y,r,minTime,timeOut);

USE_EYE = 1; % 1 - left; 2-right
%USE_EYE = Eyelink('EyeAvailable')+1;
%
% function [found] = WaitUntilFound(el,x,y,r,minTime,timeOut);
%
% checks whether eye position is within radius for minTime
% returns 1, if so; 0 if timeOut is reached
%

found = 0;

t0 = GetSecs;

inRadius = 0;


enterT = inf;
while ((~found)&&((GetSecs-t0)<timeOut))
    % sample from eyelink
    sample=Eyelink( 'newfloatsampleavailable');
    
    if sample > 0
        evt = Eyelink( 'newestfloatsample');
        xE = evt.gx(USE_EYE);
        yE = evt.gy(USE_EYE);
        pupil=evt.pa(USE_EYE);

        if ((xE~=el.MISSING_DATA) && (yE~=el.MISSING_DATA) && (pupil>0))  % pupil visisible? */

            % within radius
            if ((xE-x)^2+(yE-y)^2<r^2)
               if ~inRadius
                    inRadius = 1;
                    enterT = GetSecs;                
                    
               end
            else
                inRadius = 0;
            end

        end
        
        
    end
    
    if ((GetSecs-enterT)>minTime)&&(inRadius)
        found = 1;
     
    end
    
    
    
end % while
