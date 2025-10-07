#ifndef ROBOCOMPCALIBRATION_ICE
#define ROBOCOMPCALIBRATION_ICE
module RoboCompCalibration
{
	sequence <int> img;
	sequence <float> matrix;

	interface Calibration
	{
		void sendData (string host, int w, int h, img colour, int dw, int dh, matrix depth, matrix fxfycxcy);
	};

};

#endif
