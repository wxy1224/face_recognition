#define _ITERATOR_DEBUG_LEVEL 0

#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#if _ITERATOR_DEBUG_LEVEL == 0 && _SECURE_SCL != 0 
#error _SECURE_SCL != 0 while _ITERATOR_DEBUG_LEVEL == 0 
#endif 

#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/core/core.hpp>
#include <algorithm>
#include <iostream>
#include "opencv2/face.hpp"

using namespace cv;
using namespace std;
using namespace cv::face;


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
			Mat img = imread(path, 0);
			//printf("%s",path);
			//printf("%d %d", img.cols, img.rows);
			images.push_back(imread(path, 0));
			labels.push_back(atoi(classlabel.c_str()));
		}
	}
}


int main() {
	/********************************* hard coded values **************************************/
	// ip camera video
	//CvCapture *cap1 = cvCaptureFromFile("rtsp://admin:zhengyuan123@10.168.17.8/Streaming/Channels/101");
	CvCapture *cap1 = cvCaptureFromFile("rtsp://10.168.17.125:6554/111.sdp ");
	CvCapture *cap2 = cvCaptureFromFile("rtsp://10.168.17.125:6554/111_s.sdp ");
	// size of recognized face
	int minsize = 100;
	int maxsize = 1000;
	// scale of original video
	int scale = 1;
	// path to csv file
	string fn_csv = "C:\\Users\\xiaoyan\\Documents\\test.csv";
	// path to haarcascades
	string fn_haar = "C:\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_alt.xml";

	vector<string> names = { "xiaoyan","jing","nino","jolie","obama" };

	/******************************************************************************************/

	CascadeClassifier cascade;
	cvNamedWindow("face recognition", CV_WINDOW_AUTOSIZE);
	// These vectors hold the images and corresponding labels:
	vector<Mat> images;
	vector<int> labels;
	// load haarcascades
	if (!cascade.load(fn_haar)) {
		printf("no cascade file \n");
		return -1;
	}

	// Read in the data (fails if no valid input filename is given, but you'll get an error message):
	try {
		read_csv(fn_csv, images, labels);
	}
	catch (cv::Exception& e) {
		cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
		exit(1);
	}
	int im_width = images[0].cols;
	int im_height = images[0].rows;
	printf("im_width %d im_height %d\n", im_width, im_height);
	printf("labels %d\n", labels.size());

	//Ptr<FaceRecognizer> model = createFisherFaceRecognizer();
	Ptr<FaceRecognizer> model = createLBPHFaceRecognizer();
	model->train(images, labels);

	int count = 0;
	vector<Rect> faces;

	while (cvWaitKey(10) != atoi("q"))
	{
		//printf("frame count %d\n", count);
		
		IplImage *img_v = cvQueryFrame(cap1);

		if (!img_v) {
			img_v = cvQueryFrame(cap2);
			printf("using cap2\n");
			if (!img_v) {
				count++;
				printf("losing connection %d frame\n", count);
				if (count > 10) {
					printf("no captured image\n");
					cvReleaseCapture(&cap1);
					cvReleaseCapture(&cap2);
					cvDestroyWindow("face recognition");
					return -2;
				}
				continue;
			}
			else {
				count = 0;
			}
		}

		cv::Mat img = cv::cvarrToMat(img_v);	

		Mat gray, smallImg(cvRound(img.rows / scale), cvRound(img.cols /scale), CV_8UC1);
		cvtColor(img, gray, CV_BGR2GRAY);

		resize(img, smallImg, smallImg.size(), 0, 0, INTER_LINEAR);

		//equalizeHist(smallImg, smallImg);
		try
		{
			cascade.detectMultiScale(smallImg, faces, 1.1, 3, CV_HAAR_FIND_BIGGEST_OBJECT, Size(minsize, minsize));

			//printf("number: %zd face(s) are found.\n", faces.size());
			int limit = (faces.size() < 100000) ? faces.size() : 100000;
			for (int i = 0; i < limit; i++) {
				Rect face_i = faces[i];
				Mat face = gray(face_i);
				Mat face_resized;
				cv::resize(face, face_resized, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);
				int prediction = -1;
				double confidence = 0.0;
				model->predict(face_resized,prediction,confidence);
				rectangle(smallImg, face_i, CV_RGB(0, 255, 0), 1);
				if (confidence < 5000) {
					string box_text = format("%s, distance = %f", names[prediction], confidence);
					int pos_x = std::max(face_i.tl().x - 10, 0);
					int pos_y = std::max(face_i.tl().y - 10, 0);
					putText(smallImg, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 0), 2.0);
				}
				
				/*Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);
				ellipse(smallImg, center, Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar(100, 0, 255), 4, 8, 0);*/
				//printf("width %d\n", faces[i].width);
			}
		}
		catch (Exception ex)
		{
			printf("error in cascade\n");
		}
		imshow("img", smallImg);
		//waitKey(0);
	}

	cvReleaseCapture(&cap1);
	cvReleaseCapture(&cap2);
	cvDestroyWindow("face recognition");

	return 0;
}
