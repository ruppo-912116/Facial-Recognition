#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
	std::ifstream file(filename.c_str(), ifstream::in);
	if (!file) {
		string error_message = "No valid input file was given, please check the given filename.";
		CV_Error(CV_StsBadArg, error_message);
	}
	string line, path, classlabel;
	while (getline(file, line)) {
		stringstream liness(line);
		getline(liness, path, separator);
		getline(liness, classlabel);
		if (!path.empty() && !classlabel.empty()) {
			//Mat m = imread(path, 0);
			//Mat m2;
			//cvtColor(m, m2, CV_BGR2GRAY);
			//images.push_back(m2);
			Mat img = imread(path, IMREAD_GRAYSCALE);
			if (img.empty()) {
				cout << path << " could not be read !" << endl;
				exit(-1);
			}
			images.push_back(img);
			labels.push_back(atoi(classlabel.c_str()));
		}
	}
}

int main(int argc, const char *argv[]) {
	// Check for valid command line arguments, print usage
	// if no arguments were given.
	if (argc != 4) {
		cout << "no of counts:" << argc;
		cout << "usage: " << argv[0] << " </path/to/haar_cascade> </path/to/csv.ext> </path/to/device id>" << endl;
		cout << "\t </path/to/haar_cascade> -- Path to the Haar Cascade for face detection." << endl;
		cout << "\t </path/to/csv.ext> -- Path to the CSV file with the face database." << endl;
		cout << "\t <device id> -- The webcam device id to grab frames from." << endl;
		exit(1);
	}
	// Get the path to your CSV:
	string fn_haar = string(argv[1]);
	string fn_csv = string(argv[2]);
	int deviceId = atoi(argv[3]);
	// These vectors hold the images and corresponding labels:
	vector<Mat> images;
	vector<int> labels;
	// Read in the data (fails if no valid input filename is given, but you'll get an error message):
	try {
		read_csv(fn_csv, images, labels);
	}
	catch (cv::Exception& e) {
		cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
		// nothing more we can do
		exit(1);
	}
	// Get the height from the first image. We'll need this
	// later in code to reshape the images to their original
	// size AND we need to reshape incoming faces to this size:
	int im_width = images[0].cols;
	int im_height = images[0].rows;
	// Create a FaceRecognizer and train it on the given images:
	Ptr<FaceRecognizer> model = createFisherFaceRecognizer();
	model->train(images, labels);
	//Now we save the training data
	model->save("D:/Fisherface_Database/Fisherfacetraining.yml");
	cout << "Training completed....."<<endl;
	cout << "The stored yml file is named as Fisherfacetraining.yml";
	// That's it for learning the Face Recognition model. You now
	// need to create the classifier for the task of Face Detection.
	// We are going to use the haar cascade you have specified in the
	// command line arguments:
	//
	CascadeClassifier haar_cascade;
	haar_cascade.load(fn_haar);
	// Get a handle to the Video device:
	VideoCapture cap(deviceId);
	// Check if we can use this device at all:
	if (!cap.isOpened()) {
		cerr << "Capture Device ID " << deviceId << "cannot be opened." << endl;
		return -1;
	}
	// Holds the current frame from the Video device:
	Mat frame;
	for (;;) {
		cap >> frame;
		// Clone the current frame:
		Mat original = frame.clone();
		// Convert the current frame to grayscale:
		Mat gray;
		cvtColor(original, gray, CV_BGR2GRAY);
		// Find the faces in the frame:
		vector< Rect_<int> > faces;
		haar_cascade.detectMultiScale(gray, faces);
		// At this point you have the position of the faces in
		// faces. Now we'll get the faces, make a prediction and
		// annotate it in the video. Cool or what?
		for (int i = 0; i < faces.size(); i++) {
			// Process face by face:
			Rect face_i = faces[i];
			// Crop the face from the image. So simple with OpenCV C++:
			Mat face = gray(face_i);
			// Resizing the face is necessary for Eigenfaces and Fisherfaces. You can easily
			// verify this, by reading through the face recognition tutorial coming with OpenCV.
			// Resizing IS NOT NEEDED for Local Binary Patterns Histograms, so preparing the
			// input data really depends on the algorithm used.

			Mat face_resized;
			cv::resize(face, face_resized, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);
			// Now perform the prediction, see how easy that is:
			int prediction = model->predict(face_resized);
			int label = -1; double confidence = 0;
			model->predict(face_resized, label, confidence);
			cout << "Confidence = " << confidence << " label = " << label << endl;
			// And finally write all we've found out to the original image!
			// First of all draw a green rectangle around the detected face:
			rectangle(original, face_i, CV_RGB(0, 255, 0), 1);
			//we write the tag name here
			string Pname = "Detected";
			if (label == 0)
			{
				Pname = "Rupan";
			}
			if (label == 1)
			{
				Pname = "Pawan";
			} 
			if (label == 2)
			{
				Pname = "Ankit";
			}
			if (label == 3)
			{
				Pname = "Saroj_sir";
			}
			// Create the text we will annotate the box with:
			//string box_text = format("Prediction = %d", prediction);
			// Calculate the position for annotated text (make sure we don't
			// put illegal values in there):
			int pos_x = std::max(face_i.tl().x - 10, 0);
			int pos_y = std::max(face_i.tl().y - 10, 0);
			// And now put it into the image:
			putText(original,"Person = " + Pname, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1, CV_RGB(0, 255, 0), 2);
		}
		// Show the result:
		imshow("face_recognizer", original);
		// And display it:
		char key = (char)waitKey(20);
		// Exit this loop on escape:
		if (key == 27)
			break;
	}
	return 0;
}