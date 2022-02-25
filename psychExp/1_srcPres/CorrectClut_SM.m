function [newClutTable, oldClutTable, gamma] = CorrectClut_SM(ScreenNumber, MonitorLumTable, Indices, PlotOn, gamma_given)
% FUNCTION = CorrectClut(ScreenNumber, MonitorLumTable, Indices, PlotOn)
%
% With this function, you can linearize your screen by adjusting the Color Look Up Table
%
% MonitorLumTable = a 256 by 3 column (R G B) table of the luminance values measured by a
% spectrometer with RGB values from 0 to 255. 
% If you are only using black&white presentation, then one column
% is also allowed. Otherwise each column should contain equal numbers!
%
% Indices = a 256 by 1 column table and represent the RGB values at which the
% luminance is measured
%
% If you leave MonitorLumTable empty, the Clut will be normal
%
% Indices and MonitorLumTable should have the same column length!
%
% The given luminance values should be monotonically increasing if this function is used on a windows machine!
%
% Unfilled values (NaN) are automatically extrapolated and/or interpolated
%
% if gamma is already calculated, use this one

%% Default vals

switch nargin,
    case 0
        ScreenNumber    = 1;
        MonitorLumTable = [0:1:255]';
        Indices         = [0:1:255]';
        PlotOn          = 0;
    case 1
        MonitorLumTable = [0:1:255]';
        Indices         = [0:1:255]';
        PlotOn          = 0;
    case 2
        Indices         = [0:1:255]';
        PlotOn          = 0;
    case 3
        PlotOn          = 0;
end

%% Check for table size

if size(MonitorLumTable,2) > 3 % check whether table has correct format
    MonitorLumTable = MonitorLumTable';
end

if size(Indices,2) > 3 % check whether table has correct format
    Indices = Indices';
end

if size(MonitorLumTable,2) == 1
    MonitorLumTable = repmat(MonitorLumTable,1,3);
end

if size(Indices,2) == 1
    Indices = repmat(Indices,1,3);
end

if nargin == 5 
    if size(gamma_given) == 1
        gamma_given = repmat(gamma_given,1,3);
    end
end

%% Check for any change of input

nVals       = 256;
startVal    = 0;
endVal      = 1;

if sum(MonitorLumTable(:) == Indices(:)) == size(Indices,1)*size(Indices,2) % no changes 
    
    newClutTable = normalize(MonitorLumTable,startVal,endVal,1);
    
else % change
    
    %% Check for monotonical increase
    if sum(sum((MonitorLumTable(2:end,:)-MonitorLumTable(1:end-1,:)) < 0 )) > 0 & isWin %% breaking monotonical increase
        ERROR('Values in column of MonitorLumTable do not increase monotonically');
    end

%     %% check for interopolation
%     if sum(isNaN(MonitorLumTable(:)))
%         disp('Some values are not filled. These are automatically extrapolated and/or interpolated');
%     end

    %% Linearize the luminance values
    X = Indices;
    Y = MonitorLumTable;

    newClutTable = zeros(nVals,3);

    Colors = {'r','g','b'};
    for Col = 1:3
        Xf = X(isfinite(Y(:,Col)),Col);
        Yf = Y(isfinite(Y(:,Col)),Col);

        logXf = log(Xf);
        logYf = log(Yf);
        
        % exclude very low values
        p = polyfit(log(Xf(30:end)),log(Yf(30:end)),1);
        
        if nargin == 5
            gamma = gamma_given(Col);
        else
            gamma = p(1);
        end

        bii = (Yf(end)./(Xf(end).^(1/gamma)));

        Xii = linspace(startVal,endVal,nVals);
        
        Yii = bii.*(Xii.^(1/gamma));
        nYii = normalize(Yii,startVal,endVal,2);

        newClutTable(:,Col) = nYii;
        
        if PlotOn == 1
            % plot linear gamma
            subplot(2,3,Col), plot(logXf,logYf,Colors{Col})
            xlabel('Log(X)')
            ylabel('Log(Y)')
            axis square
            hold on
            
            % fit
            bi = Yf(end)./(Xf(end).^gamma);
            Xi = linspace(startVal,endVal,nVals);
            Yi = bi.*(Xi.^gamma);

            nYi = normalize(Yi,startVal,endVal,2);

            subplot(2,3,Col+3), plot(Xf,normalize(Yf,nYi(Xf(1)),endVal,1),'r')
            hold on
            axis square
            subplot(2,3,Col+3), plot(nYi,'g')
            subplot(2,3,Col+3), plot(nYii,'b')
            
            legend()
        end
    end

end

%% Load clut

oldClutTable = Screen('LoadNormalizedGammaTable', ScreenNumber, newClutTable);
