#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/photo/photo.hpp"
#include <fstream>
#include <iostream>
#include <omp.h>

using namespace cv;
using namespace std;

Mat tempColor;
int flag[256];

void drawHist(Mat* img, bool write, bool draw)
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
		line(histImg,
			Point((i - 1) * binStep, height - cvRound(hist.at<float>(i - 1))),
			Point((i)*binStep, height - cvRound(hist.at<float>(i))),
			Scalar(0, 0, 255));

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
				if (count == 1 && flag[count] < 60)
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
	if (write)
	{
		for (int i = 0; i < count; i++)
		{
			cout << "flag" + to_string(i) + ": " + to_string(flag[i]) << endl;
		}
		cout << "count num: " + to_string(count) << endl;
	}
	if (draw)
	{
		for (int i = 0; i < histSize[0]; i++)
		{
			cout << i << " " << cvRound(hist.at<float>(i)) << endl;
		}
		imshow("window", histImg);
		waitKey(0);
		//imshow("window", *img);
		//waitKey(0);
	}
}

vector<vector<Point>> findAndDrawContour(Mat img, Mat src, string dst, bool writeFlag)

{
	// find contour
	vector<vector<Point>> contours;
	vector<vector<Point>> finalContours;
	findContours(img, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	// filter contour
	vector<RotatedRect> areas;
	for (auto contour : contours)
	{
		auto box = minAreaRect(contour);
		if (box.size.width < 2000 || box.size.height < 1000)
			continue;
		areas.push_back(box);
		finalContours.push_back(contour);
	}
	//cout << "contours num: " << finalContours.size() << endl;
	// draw contour
	Mat localTempColor;
	cvtColor(src, localTempColor, COLOR_GRAY2RGB);
	for (int i = 0; i < finalContours.size(); i++)
	{
		drawContours(localTempColor, finalContours, i, CV_RGB(255, 0, 0), 1);
	}
	if (writeFlag)
		imwrite(dst, localTempColor);

	return finalContours;
}

Mat drawLine(Vec4f lineParam1, Mat img, int mode, int thick)
{
	Mat res = img;
	if (mode == 0)
	{
		double k1 = lineParam1[1] / lineParam1[0];
		double b1 = lineParam1[3] - k1 * lineParam1[2];
		int x0 = 0, x1 = img.cols;
		int y0 = (int)(k1 * x0 + b1);
		int y1 = (int)(k1 * x1 + b1);
		line(res, Point(x0, y0), Point(x1, y1), CV_RGB(255, 0, 255), thick);
	}

	if (mode == 1)
	{
		double k1 = lineParam1[0] / lineParam1[1];
		double b1 = lineParam1[2] - k1 * lineParam1[3];
		int y0 = 0, y1 = img.rows;
		int x0 = (int)(k1 * y0 + b1);
		int x1 = (int)(k1 * y1 + b1);
		line(res, Point(x0, y0), Point(x1, y1), CV_RGB(255, 0, 255), thick);
	}

	return res;
}

Point2f getSymmetryPoint(Vec4f lineParam, Point2f passPoint)
{
	Vec4f targetLineParam;
	targetLineParam[0] = -lineParam[1];
	targetLineParam[1] = lineParam[0];
	targetLineParam[2] = passPoint.x;
	targetLineParam[3] = passPoint.y;
	float k1, b1, k2, b2;
	k1 = lineParam[1] / lineParam[0];
	b1 = lineParam[3] - k1 * lineParam[2];
	k2 = targetLineParam[0] / targetLineParam[1];
	b2 = targetLineParam[2] - k2 * targetLineParam[3];
	float mx, my;
	my = (k1 * b2 + b1) / (1 - k1 * k2);
	mx = k2 * my + b2;
	float x, y;
	x = 2 * mx - passPoint.x;
	y = 2 * my - passPoint.y;
	return Point2f(x, y);
}

int main()
{
	bool write = false;

	int64 start, end;
	Mat img = imread("./Origin/0/463990d5-7b7e-4e0b-8979-5f86564c0b63_camera0-type5.bmp", IMREAD_GRAYSCALE);
	Mat img_c = imread("./Origin/0/463990d5-7b7e-4e0b-8979-5f86564c0b63_camera0-type5.bmp", IMREAD_COLOR);
	Mat img_o = imread("./Origin/0/463990d5-7b7e-4e0b-8979-5f86564c0b63_camera0-type5.bmp", IMREAD_COLOR);
	
	start = getTickCount();
	
	// thresh
	Mat thresh;
	drawHist(&img, false, false);
	threshold(img, thresh, flag[1], 255, THRESH_BINARY_INV);
	if (write)
		imwrite("./test/thresh.jpg", thresh);

	// contour
	vector<vector<Point>> totalContour = findAndDrawContour(thresh, img, "./test/contour.jpg", write);

	// roi
	auto box = boundingRect(totalContour[0]);
	if (write)
	{
		rectangle(img_c, box, Scalar(255, 0, 255), 5);
		imwrite("./test/boundingBox.jpg", img_c);
	}
	int roi_width = 250;  //300
	int roi_height = 430;  //450
	int roi2_width = 400;  //240
	int roi2_height = 350;  //350
	Rect roi = Rect(box.br().x - roi_width, box.tl().y, roi_width, roi_height-40);
	Rect roi2 = Rect(box.br().x - roi2_width, box.br().y - roi2_height, roi2_width, roi2_height);
	Rect roi_line1 = Rect(box.tl().x, box.tl().y, box.br().x - box.tl().x - 2000, 500);
	Rect roi_line3 = Rect(box.tl().x, box.br().y - 500, box.br().x - box.tl().x - 2000, 500);
	if (write)
	{
		//rectangle(img_c, roi, Scalar(255, 0, 255), 5);
		//rectangle(img_c, roi2, Scalar(255, 0, 255), 5);
		rectangle(img_c, roi_line1, Scalar(255, 0, 255), 5);
		rectangle(img_c, roi_line3, Scalar(255, 0, 255), 5);
		imwrite("./test/roi.jpg", img_c);
	}

	// draw line
	int x, y;
	int px = 0, py = 0;
	int index, nextIndex;
	vector<Point> linePoints;
	vector<Point> LinePoints1, LinePoints3;
	for (int i = box.tl().x; i < box.br().x - 20; i++)
	{
		x = i;
		y = box.tl().y;
		index = y * img.cols + x;
		nextIndex = (y + 1) * img.cols + x;
		while (img.data[index] > 250)
		{
			y++;
			index = y * img.cols + x;
			nextIndex = (y + 1) * img.cols + x;
		}
		LinePoints1.push_back(Point(x, y));
		if (i > box.tl().x && write)
		{
			line(img_c, Point(px, py), Point(x, y), CV_RGB(0, 255, 255), 2);
		}
		px = x;
		py = y;
	}
	for (int i = box.tl().y + 300; i < box.br().y - 2; i++)
	{
		x = box.br().x;
		y = i;
		index = y * img.cols + x;
		nextIndex = y * img.cols + (x - 1);
		while (img.data[index] > 250)
		{
			x--;
			index = y * img.cols + x;
			nextIndex = y * img.cols + (x - 1);
		}
		linePoints.push_back(Point(x, y));
		if (write)
		{
			line(img_c, Point(px, py), Point(x, y), CV_RGB(0, 255, 255), 2);
		}
		px = x;
		py = y;
	}
	for (int i = box.tl().x; i < box.br().x - 20; i++)
	{
		x = i;
		y = box.br().y;
		index = y * img.cols + x;
		nextIndex = (y - 1) * img.cols + x;
		while (img.data[index] > 250)
		{
			y--;
			index = y * img.cols + x;
			nextIndex = (y - 1) * img.cols + x;
		}
		LinePoints3.push_back(Point(x, y));
		if (i > box.tl().x && write)
		{
			line(img_c, Point(px, py), Point(x, y), CV_RGB(0, 255, 255), 2);
		}
		px = x;
		py = y;
	}
	if (write)
		imwrite("./test/drawLine.jpg", img_c);
	//write = false;

	// fit arc
	vector<Point> points;
	vector<Point> points2;
	//for (Point contourPoint : totalContour[0])
	//{
	//	if (roi2.contains(contourPoint))
	//		points2.push_back(contourPoint);
	//}
	for (int i = 0; i < linePoints.size(); i++)
	{
		if (roi.contains(linePoints[i]))
			points.push_back(linePoints[i]);
	}
	for (int i = 0; i < linePoints.size(); i++)
	{
		if (roi2.contains(linePoints[i]))
			points2.push_back(linePoints[i]);
	}
	//RotatedRect arc = fitEllipse(points);
	RotatedRect arc2 = fitEllipse(points2);
	if (write)
	{
		//ellipse(img_c, arc, Scalar(0, 255, 255), 2);
		ellipse(img_c, arc2, Scalar(0, 255, 255), 2);
		imwrite("./test/fit.jpg", img_c);
	}

	// fit line
	vector<Point> pointLines1;
	vector<Point> pointLines3;
	for (int i = 0; i < LinePoints1.size(); i++)
	{
		if (roi_line1.contains(LinePoints1[i]))
			pointLines1.push_back(LinePoints1[i]);
	}
	for (int i = 0; i < LinePoints3.size(); i++)
	{
		if (roi_line3.contains(LinePoints3[i]))
			pointLines3.push_back(LinePoints3[i]);
	}
	Vec4f lineParam1, lineParam2, lineParam3;
	fitLine(pointLines1, lineParam1, DIST_L2, 0, 1e-2, 1e-2);
	fitLine(pointLines3, lineParam3, DIST_L2, 0, 1e-2, 1e-2);
	Vec4f lineParamMid;
	lineParamMid[0] = (lineParam1[0] + lineParam3[0]) / 2;
	lineParamMid[1] = (lineParam1[1] + lineParam3[1]) / 2;
	lineParamMid[2] = (lineParam1[2] + lineParam3[2]) / 2;
	lineParamMid[3] = (lineParam1[3] + lineParam3[3]) / 2;
	Mat lineResult = img_c;
	if (write)
	{
		cout << lineParam1 << lineParam3 << endl;
		lineResult = drawLine(lineParam1, lineResult, 0, 2);
		lineResult = drawLine(lineParam3, lineResult, 0, 2);
		lineResult = drawLine(lineParamMid, lineResult, 0, 2);
		imwrite("./test/fitline.jpg", lineResult);
	}
	Point2f symRotatedRec0, symRotatedRec1, symRotatedRec2;
	Point2f originalRotatedRec[4];
	arc2.points(originalRotatedRec);
	symRotatedRec0 = getSymmetryPoint(lineParamMid, originalRotatedRec[0]);
	symRotatedRec1 = getSymmetryPoint(lineParamMid, originalRotatedRec[1]);
	symRotatedRec2 = getSymmetryPoint(lineParamMid, originalRotatedRec[2]);
	RotatedRect symRotatedRec = RotatedRect(symRotatedRec0, symRotatedRec1, symRotatedRec2);
	ellipse(img_c, symRotatedRec, Scalar(0, 0, 0), 2);
	if (write)
		imwrite("./test/symmetry.jpg", img_c);

	Mat finalImg, finalThresh;
	cvtColor(img_c, finalImg, COLOR_RGB2GRAY);
	threshold(finalImg, finalThresh, 250, 255, THRESH_BINARY);
	if (write)
		imwrite("./test/thresh.jpg", finalThresh);
	vector<vector<Point>> finalContours;
	findContours(finalThresh, finalContours, RETR_LIST, CHAIN_APPROX_SIMPLE);
	for (auto contour : finalContours)
	{
		Rect box = boundingRect(contour);
		if (box.width > 100 || box.height > 100 || box.width < 10 || box.height < 10)
			continue;
		rectangle(img_o, box, Scalar(255, 0, 0), 2);
	}
	if (true)
		imwrite("./test/locate.jpg",img_o);

	end = getTickCount();
	cout << "time: " << (end - start) / getTickFrequency() << endl;

	return 0;
}