function [] = initEL1000SM(filename,ScrRes,ScrDist,ScrWidth,ScrHeight)

switch nargin,
    case 0
        filename = 'testfile';
        ScrRes = [1280 1024];
        ScrDist = 730;
        ScrWidth = 400;
        ScrHeight = 300;
    case 1
        ScrRes = [1280 1024];
        ScrDist = 730;
        ScrWidth = 400;
        ScrHeight = 300;
    case 2
        ScrDist = 730;
        ScrWidth = 400;
        ScrHeight = 300;
    case 3
        ScrWidth = 400;
        ScrHeight = 300;
    case 4
        ScrHeight = 300;
end
        
% open link
if ~Eyelink('isconnected');
    Eyelink('initialize');
end


status = Eyelink('openfile', filename);
if status~=0
	Eyelink('Shutdown');
    clear mex;
	error(sprintf('Cannot create %s (error: %d)- Eyelink shutdown', status,filename));
end


% send standard parameters
Eyelink('command', ['add_file_preamble_text ','EL1000, visual search, wet, original name wet']);
Eyelink('command', 'calibration_type = HV13');
Eyelink('command', 'saccade_velocity_threshold = 35');
Eyelink('command', 'saccade_acceleration_threshold = 9500');
Eyelink('command', 'file_event_filter = LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON,INPUT');
Eyelink('command', 'link_event_filter = LEFT,RIGHT,FIXATION,BUTTON');
Eyelink('command', 'link_sample_data  = LEFT,RIGHT,GAZE,AREA');
Eyelink('command', 'button_function 5 ''accept_target_fixation''');

% monocular parameters
Eyelink('command', 'binocular_enabled = NO');
Eyelink('command', 'active_eye = LEFT');
Eyelink('command', 'select_eye_after_validation = YES'); % select the best eye after calibration
Eyelink('command', 'pupil_size_diameter = DIAMETER');
Eyelink('command', 'cal_repeat_first_target = YES');
Eyelink('command', 'val_repeat_first_target = YES');
Eyelink('command', 'corneal_mode = YES'); % 
Eyelink('command', 'use_high_speed = YES'); % higher sampling rate
Eyelink('command', 'pupil_crosstalk_fixup = 0.000, -0.001'); % correct for pupil size invariances depending on eye position (1-300 deg units)
Eyelink('command', ['screen_pixel_coords = 0.0, 0.0, ' num2str(ScrRes(1)) '.0, ' num2str(ScrRes(2)) '.0']); % 
Eyelink('command', 'calibration_area_proportion = 0.88, 0.88'); % 
Eyelink('command', 'validation_area_proportion = 0.88, 0.88'); % 
% make driftcorrection possible
Eyelink('command', 'driftcorrect_cr_disable = OFF');

if ( length(num2str(ScrDist)) < 3 ) %% if in centimeters .. then make it millimeters
    ScrDist = [num2str(ScrDist) regexprep(num2str(zeros(1,3-length(num2str(ScrDist)))),'\s','')];
end

if ( length(num2str(ScrWidth)) < 3 ) %% if in centimeters .. then make it millimeters
    ScrWidth = [num2str(ScrWidth) regexprep(num2str(zeros(1,3-length(num2str(ScrWidth)))),'\s','')];
end

if ( length(num2str(ScrWidth)) < 3 ) %% if in centimeters .. then make it millimeters
    ScrWidth = [num2str(ScrWidth) regexprep(num2str(zeros(1,3-length(num2str(ScrWidth)))),'\s','')];
end

Eyelink('command', ['screen_distance = ' num2str(ScrDist+20) ' ' num2str(ScrDist+50)]); % top and bottom distance
Eyelink('command', ['screen_phys_coords = ' num2str(-1*ScrWidth/2) '.0, ' num2str(ScrHeight/2) '.0, ' num2str(ScrWidth/2) '.0, ' num2str(-1*ScrHeight/2) '.0']); % width of the screen you are using
    
Eyelink('command', 'sample_rate = 1000'); % 