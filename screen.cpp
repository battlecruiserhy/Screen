#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <fstream>
#include <iostream>

using namespace cv;
using namespace std;

Mat tempColor;
int flag[256];

void drawHist(Mat* img)
{
	int numbins = 256;
	int channels[1] = { 0 };
	int histSize[1] = { 256 };
	float range[] = { 0,256 };
	const float* histRanges[] = { range };
	int count = 0;
	int min = 9999;
	int local_flag;
	int state = 1; //up

	Mat hist;
	calcHist(img, 1, channels, Mat(), hist, 1, histSize, histRanges);

	int width = 512;
	int height = 300;

	Mat histImg(width, height, CV_8UC3, Scalar(0, 0, 0));
	normalize(hist, hist, 0, height, NORM_MINMAX);

	int binStep = cvRound((float)width / (float)histSize[0]);

	for (int i = 1; i < histSize[0]; i++)
	{
		//cout << "hist"+to_string(i-1)+": " << endl;
		//cout << cvRound(hist.at<float>(i - 1)) << endl;
		line(histImg,
			Point((i - 1) * binStep, height - cvRound(hist.at<float>(i - 1))),
			Point((i)*binStep, height - cvRound(hist.at<float>(i))),
			Scalar(0, 0, 255));
		//if (cvRound(hist.at<float>(i - 1)) != 0 && cvRound(hist.at<float>(i))==0)
		//{
		//	flag[count] = i;
		//	count++;
		//}


		if (cvRound(hist.at<float>(i - 1)) <= min)
		{
			min = cvRound(hist.at<float>(i - 1));
			local_flag = i - 1;
			if (cvRound(hist.at<float>(i)) > min && state == 0 && cvRound(hist.at<float>(i - 1)) < 3)
			{
				if (count == 0)
				{
					while (cvRound(hist.at<float>(local_flag)) == cvRound(hist.at<float>(local_flag - 1)))
					{
						local_flag--;
					}
				}
				min = 9999;
				flag[count] = local_flag;
				if (count == 1 && flag[count] < 50)
				{
					count--;
				}
				count++;
			}
		}
		if (cvRound(hist.at<float>(i - 1)) > cvRound(hist.at<float>(i)))
			state = 0;	//down
		if (cvRound(hist.at<float>(i - 1)) < cvRound(hist.at<float>(i)))
			state = 1;	//up

	}
	for (int i = 0; i < count; i++)
	{
		cout << "flag" + to_string(i) + ": " + to_string(flag[i]) << endl;
	}
	cout << "count num: " + to_string(count) << endl;
	//imshow("window", histImg);
	//waitKey(0);
	//imshow("window", *img);
	//waitKey(0);
}

int calcContour(vector<vector<Point>> foundContours, Mat r)
{
	int width = r.cols;
	int height = r.rows;
	int* contourTableY = new int[height];
	int* contourTableX = new int[width];
	for (int i = 0; i < height; i++)
		contourTableY[i] = 0;
	for (int i = 0; i < width; i++)
		contourTableX[i] = 0;
	// build table
	for (int i = 0; i < foundContours.size(); i++)
	{
		for (int j = 0; j < foundContours[i].size(); j++)
		{
			contourTableY[foundContours[i][j].y]++;
			contourTableX[foundContours[i][j].x]++;
		}
	}
	//fstream f("calcContour.txt", ios::out);
	//f << "contourTalbeY" << endl;
	cout << "findy: " << endl;
	for (int i = 1; i < height; i++)
		if (contourTableY[i - 1] > 10 && contourTableY[i] <= 2)
		{
			cout << i << endl;
			return i;
		}
	//f << contourTableY[i] << endl;
//f << "contourTableX" << endl;
	for (int i = 0; i < width; i++)
	{

	}
	//f << contourTableX[i] << endl;
//f.close();


}

void getOilArea()
{
	// sub
	//Mat r2 = imread("./Origin/2/2.jpg", 0);
	//Mat r1 = imread("./Origin/2/1.jpg", 0);
	Mat r2 = imread("./Origin2/0/2.jpg", 0);
	Mat r1 = imread("./Origin2/0/1.jpg", 0);
	/*Mat thresRes;
	threshold(r1, thresRes, 40, 255, THRESH_BINARY);
	Mat subRes = r2 - thresRes;*/
	double start, end;
	start = getTickCount();
	// preprocess
	medianBlur(r1, r1, 7);
	medianBlur(r2, r2, 7);
	Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
	Mat temp1, temp2;
	erode(r1, temp1, kernel, Point(-1, -1), 5);
	dilate(temp1, temp1, kernel, Point(-1, -1), 5);
	erode(r2, temp2, kernel, Point(-1, -1), 5);
	dilate(temp2, temp2, kernel, Point(-1, -1), 5);
	//imwrite("test.jpg", temp1);

	Mat subRes = temp2 - temp1;
	//imwrite("./result/subRes.jpg", subRes);
	//imshow("window", subRes);
	//waitKey(0);

	// threshhold
	Mat thresResult, thresResult2;
	drawHist(&subRes);
	threshold(subRes, thresResult, flag[1], 255, THRESH_TOZERO_INV);		//flag[1]
	//drawHist(&thresResult);
	threshold(thresResult, thresResult2, flag[0], 255, THRESH_TOZERO);		//flag[0]
	//drawHist(&thresResult2);
	//threshold(thresResult, thresResult2, 90, 255, THRESH_BINARY);
	//imwrite("./result/thresResult2.jpg", thresResult2);
	//imshow("window", thresResult2);
	//imwrite("thresr1_thresResRes.jpg", thresResult2);
	//waitKey(0);

	// find contour
	vector<vector<Point>> contours;
	vector<vector<Point>> finalContours;
	findContours(thresResult2, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	//findContours(thresResult2, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
	vector<RotatedRect> areas;
	for (auto contour : contours)
	{
		auto box = minAreaRect(contour);
		//if (box.size.width < 2000 || box.size.height < 500)
		if (box.size.height < 3000)		// use height*width will be better
			continue;
		areas.push_back(box);
		finalContours.push_back(contour);
	}
	cout << finalContours.size() << endl;
	end = getTickCount();
	cout << "handle time: " << (end - start) / getTickFrequency() << endl;
	// process contour
	int replaceY = calcContour(finalContours, temp2);

	// replace contour
	vector<vector<Point>> contours2;
	vector<vector<Point>> finalContours2;
	Mat r2r = imread("./Origin2/0/2.jpg", 0);

	start = getTickCount();
	threshold(r2r, r2r, 10, 255, THRESH_BINARY);
	findContours(r2r, contours2, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	vector<RotatedRect> areas2;
	for (auto contour : contours2)
	{
		auto box = minAreaRect(contour);
		if (box.size.width < 2000 || box.size.height < 500)
			continue;
		areas2.push_back(box);
		finalContours2.push_back(contour);
	}
	cvtColor(r2, tempColor, COLOR_GRAY2RGB);
	for (int i = 0; i < finalContours2.size(); i++)
	{
		drawContours(tempColor, finalContours2, i, CV_RGB(255, 0, 0));
	}
	end = getTickCount();
	cout << "raw contour time: " << (end - start) / getTickFrequency() << endl;
	imwrite("./result/test.jpg", tempColor);


	cout << finalContours2.size() << endl;
	for (int i = 0; i < finalContours2.size(); i++)
	{
		for (int j = 0; j < finalContours2[i].size(); j++)
		{
			if (finalContours2[i][j].y > replaceY)
			{
				for (int k = 0; k < finalContours[0].size(); k++)
				{
					int f1x = finalContours[0][k].x;
					int f2x = finalContours2[i][j].x;
					int f1y = finalContours[0][k].y;
					int f2y = finalContours2[i][j].y;
					if (f1x == f2x && f1y > replaceY)
					{
						//cout << "b:" << f1y << ' ' << f2y << endl;
						finalContours[0][k].y = finalContours2[i][j].y;
						//cout << "a: " << finalContours[0][k].y << ' ' << finalContours2[i][j].y << endl;
					}
				}
			}
		}
	}


	// draw conotur
	Mat thresColor;
	cvtColor(thresResult2, thresColor, COLOR_GRAY2RGB);
	//Mat tempColor;
	cvtColor(r2, tempColor, COLOR_GRAY2RGB);
	for (int i = 0; i < finalContours.size(); i++)
	{
		drawContours(tempColor, finalContours, i, CV_RGB(255, 0, 0));
	}

	//imshow("window", tempColor);
	//waitKey(0);

	imwrite("./result/try_real_oil.jpg", tempColor);
}

int main()
{
	namedWindow("window", cv::WINDOW_NORMAL);
	resizeWindow("window", cv::Size(640, 720));	

	double start, end;
	start = getTickCount();
	getOilArea();
	end = getTickCount();
	cout << "total time: " << (end - start) / getTickFrequency() << endl;
	return 0;
}