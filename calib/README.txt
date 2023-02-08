These mat files are created by loading the original mats supplied by the Katwijk dataset. However, those mats cannot be read by scipy.io.loadmat because they are MATLAB objects - so, we convert them as structs and resave. The commands looks something like

>> load('LocCam_calibration.mat');
>> stereoParams = struct(stereoParams);
>> stereoParams.CameraParameters1 = struct(stereoParams.CameraParameters1);
>> stereoParams.CameraParameters2 = struct(stereoParams.CameraParameters2);
>> save('LocCam_calibstruct.mat');
