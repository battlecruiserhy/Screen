#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <fstream>
#include <iostream>

#include <omp.h>

using namespace cv;
using namespace std;

Mat tempColor;
int flag[256];

void combineImages()
{
	// get imgs
	string path[3];
	path[0] = "./Origin/0/2.jpg";
	path[1] = "./Origin/1/2.jpg";
	path[2] = "./Origin/2/2.jpg";
	vector<Mat> imgSet;
	for (int i = 0; i < 3; i++)
	{
		Mat img = imread(path[i], 0);
		imgSet.push_back(img);
	}
	//combine imgs
	Mat combineImg, tempImg;
	hconcat(imgSet[0], imgSet[1], tempImg);
	hconcat(tempImg, imgSet[2], combineImg);
	imwrite("r2_real.jpg", combineImg);
}

void drawHist(Mat* img, bool write, bool draw)
{
	int numbins = 256;
	int channels[1] = { 0 };
	int histSize[1] = { 256 };
	float range[] = { 0,256 };
	const float* histRanges[] = {range};
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
			min = cvRound(hist.at<float>(i-1));
			local_flag = i-1;
			if (cvRound(hist.at<float>(i)) > min && state == 0 && cvRound(hist.at<float>(i - 1)) < 3)
			{
				if (count == 0)
				{
					while (cvRound(hist.at<float>(local_flag)) == cvRound(hist.at<float>(local_flag-1)))
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

int calcContour(vector<Point> targetContour, Mat r)
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
	for (int i = 0; i < targetContour.size(); i++)
	{
		contourTableY[targetContour[i].y]++;
		contourTableX[targetContour[i].x]++;
	}

	fstream f("calcContour.txt", ios::out);
	f << "contourTalbeY" << endl;
	//cout << "findy: " << endl;
	for (int i = 1; i < height; i++)
	{
		if (contourTableY[i - 1] > 10 && contourTableY[i] <= 2)
		{
			//cout << i << endl;
			//return i;
		}
		f << contourTableY[i] << endl;
	}
	f << "contourTableX" << endl;
	for (int i = 0; i < width; i++)
	{
		f << contourTableX[i] << endl;
	}
	f.close();
	delete[]contourTableY;
	delete[]contourTableX;
	contourTableY = nullptr;
	contourTableX = nullptr;
	return 0;
}

vector<vector<Point>> findAndDrawContour(Mat img, Mat src, string dst, bool writeFlag)
{
	// find contour
	vector<vector<Point>> contours;
	vector<vector<Point>> finalContours;
	findContours(img, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	// filter contour
	vector<RotatedRect> areas;
	for (auto contour : contours)
	{
		auto box = minAreaRect(contour);
		if (box.size.width < 2000 || box.size.height < 2000)
			continue;
		areas.push_back(box);
		finalContours.push_back(contour);
	}
	//cout << "contours num: " << finalContours.size() << endl;
	//draw contour
	Mat localTempColor;
	cvtColor(src, localTempColor, COLOR_GRAY2RGB);
	for (int i = 0; i < finalContours.size(); i++)
	{
		drawContours(localTempColor, finalContours, i, CV_RGB(255, 0, 0), 5);
	}
	if (writeFlag == true)
		imwrite(dst, localTempColor);

	return finalContours;
}

Mat pixelwiseScan(Mat sourceImg, Rect roi, bool write)
{
	int start_x = roi.tl().x;
	int start_y = roi.tl().y;
	int x_length = roi.width;
	int y_length = roi.height;

	int height;
	int width;
	//int index, aboveIndex, frontIndex, backIndex, belowIndex;
	int index, belowIndex;
	int score = 0;
	int fillColor = 0;
	int fillThreshold = 30;
	bool continuousWhite = false;
	int tempIndex;

	//Mat modifyImg1 = sourceImg.clone();
	Mat modifyImg1 = sourceImg;
	height = modifyImg1.rows;
	width = modifyImg1.cols;
	score = 0;
	fillColor = 0;
	fillThreshold = 30;
#pragma omp parallel for
	for (int i = start_x + 1; i < start_x + x_length - 1; i++)
	{
		score = 0;
		for (int j = start_y + 1; j < start_y + y_length - 1; j++)
		{
			index = j * width + i;
			if (modifyImg1.data[index] > 0)
			{
				//aboveIndex = (j - 1) * width + i;
				belowIndex = (j + 1) * width + i;
				//frontIndex = j * width + (i - 1);
				//backIndex = j * width + (i + 1);
				score++;
				if (modifyImg1.data[belowIndex] == 0)
				{
					if (score < fillThreshold)
					{
						while (score > 0)
						{
							tempIndex = (j - score + 1) * width + i;
							modifyImg1.data[tempIndex] = 0;
							score--;
						}
					}
					else
						score = 0;
				}
			}
		}
	}
	if (write == true)
		imwrite("./test2/modifyImg1.jpg", modifyImg1);

	// vertical scan(left to right)
	//Mat modifyImg2 = modifyImg1.clone();
	Mat modifyImg2 = modifyImg1;
	height = modifyImg2.rows;
	width = modifyImg2.cols;
	score = 0;
	fillColor = 0;
	fillThreshold = 30;  //5
	for (int i = start_y + 1; i < start_y + y_length - 1; i++)
	{
		score = 0;
		for (int j = start_x + 1; j < start_x + x_length - 1; j++)
		{
			index = i * width + j;
			if (modifyImg2.data[index] > 0)
			{
				//aboveIndex = (j - 1) * width + i;
				belowIndex = index + 1;
				//frontIndex = j * width + (i - 1);
				//backIndex = j * width + (i + 1);
				score++;
				if (modifyImg2.data[belowIndex] == 0)
				{
					if (score < fillThreshold)
					{
						while (score > 0)
						{
							modifyImg2.data[index - score + 1] = 0;
							score--;
						}
					}
					else
						score = 0;
				}
			}
		}
	}
	if (write == true)
		imwrite("./test2/modifyImg2.jpg", modifyImg2);

	return modifyImg2;
}

Mat pixelwiseRefine(Mat sourceImg, Rect roi, bool write)
{
	int start_x = roi.tl().x;
	int start_y = roi.tl().y;
	int x_length = roi.width;
	int y_length = roi.height;

	int height;
	int width;
	//int index, aboveIndex, frontIndex, backIndex, belowIndex;
	int index, belowIndex;
	int score = 0;
	int fillColor = 255;
	int fillThreshold = 10;
	bool continuousWhite = false;
	int tempIndex;

	//Mat modifyImg1 = sourceImg.clone();
	Mat modifyImg1 = sourceImg;
	height = modifyImg1.rows;
	width = modifyImg1.cols;
	score = 0;
	fillColor = 255;
	fillThreshold = 10;
#pragma omp parallel for
	for (int i = start_x + 1; i < start_x + x_length - 1; i++)
	{
		score = 0;
		for (int j = start_y + 1; j < start_y + y_length - 1; j++)
		{
			index = j * width + i;
			if (modifyImg1.data[index] == 0)
			{
				//aboveIndex = (j - 1) * width + i;
				belowIndex = (j + 1) * width + i;
				//frontIndex = j * width + (i - 1);
				//backIndex = j * width + (i + 1);
				score++;
				if (modifyImg1.data[belowIndex] > 0)
				{
					if (score < fillThreshold)
					{
						while (score > 0)
						{
							tempIndex = (j - score + 1) * width + i;
							modifyImg1.data[tempIndex] = 255;
							score--;
						}
					}
					else
						score = 0;
				}
			}
		}
	}
	if (write == true)
		imwrite("./test2/modifyImg1.jpg", modifyImg1);

	// vertical scan(left to right)
	//Mat modifyImg2 = modifyImg1.clone();
	Mat modifyImg2 = modifyImg1;
	height = modifyImg2.rows;
	width = modifyImg2.cols;
	score = 0;
	fillColor = 255;
	fillThreshold = 10;  //5
	for (int i = start_y + 1; i < start_y + y_length - 1; i++)
	{
		score = 0;
		for (int j = start_x + 1; j < start_x + x_length - 1; j++)
		{
			index = i * width + j;
			if (modifyImg2.data[index] == 0)
			{
				//aboveIndex = (j - 1) * width + i;
				belowIndex = index + 1;
				//frontIndex = j * width + (i - 1);
				//backIndex = j * width + (i + 1);
				score++;
				if (modifyImg2.data[belowIndex] > 0)
				{
					if (score < fillThreshold)
					{
						while (score > 0)
						{
							modifyImg2.data[index - score + 1] = 255;
							score--;
						}
					}
					else
						score = 0;
				}
			}
		}
	}
	if (write == true)
		imwrite("./test2/modifyImg2.jpg", modifyImg2);

	return modifyImg2;
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

void leftSide()
{
	//Mat imgR1 = imread("./Origin2/0/1.jpg", IMREAD_GRAYSCALE);
	Mat imgR2 = imread("./Origin2/0/2.jpg", IMREAD_GRAYSCALE);
	Mat imgR2_c = imread("./Origin2/0/2.jpg", IMREAD_COLOR);

	bool write = false;
	bool showHist = false;

	int64 start, end;
	int64 totalTime = 0;

	// threshold
	start = getTickCount();
	Mat thresR2;
	drawHist(&imgR2, write, showHist);
	int bias = 10;
	threshold(imgR2, thresR2, flag[1] - bias, 255, THRESH_BINARY);	//80
	//threshold(imgR2, thresR2, 0, 255, THRESH_OTSU);
	end = getTickCount();
	totalTime += end - start;
	cout << "thresh time: " << (end - start) / getTickFrequency() << endl;
	if (write)
		imwrite("./test2/threshTest.jpg", thresR2);

	// reduce ROI
	start = getTickCount();
	int roi1_width;
	int roi1_height = 1000;  //1000
	int roi2_width = 1000;
	int roi2_height;
	int roi3_width;
	int roi3_height = 1000;  //1000
	int modifyAmount = 200;
	vector<vector<Point>> contours;
	findContours(thresR2, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	// filter contour
	vector<Rect> areas;
	Rect roi1;
	Rect roi2;
	Rect roi3;
	Rect roi1_m;
	Rect roi2_m;
	Rect roi3_m;
	for (auto contour : contours)
	{
		auto box = boundingRect(contour);
		if (box.width < 2000 || box.height < 2000)
			continue;
		areas.push_back(box);
		// get ROI
		roi1_width = box.br().x - box.tl().x;
		roi2_height = box.br().y - box.tl().y;
		roi3_width = box.br().x - box.tl().x;
		//cout << box.tl() << endl;
		roi1 = Rect(box.tl().x + roi2_width, box.tl().y, roi1_width - roi2_width, roi1_height);
		roi2 = Rect(box.tl().x, box.tl().y, roi2_width, roi2_height);
		roi3 = Rect(box.br().x + roi2_width - roi3_width, box.br().y - roi3_height, roi3_width - roi2_width, roi3_height);
		roi1_m = Rect(box.tl().x + roi2_width + modifyAmount, box.tl().y, roi1_width - roi2_width - 2 * modifyAmount, roi1_height);
		roi2_m = Rect(box.tl().x, box.tl().y + roi1_height, roi2_width, roi2_height - 2 * roi3_height);
		roi3_m = Rect(box.br().x + roi2_width - roi3_width + modifyAmount, box.br().y - roi3_height, roi3_width - roi2_width - 2 * modifyAmount, roi3_height);

		if (write)
		{
			rectangle(imgR2_c, roi1_m, Scalar(0, 0, 255), 10);
			rectangle(imgR2_c, roi2_m, Scalar(0, 255, 0), 10);
			rectangle(imgR2_c, roi3_m, Scalar(255, 0, 0), 10);
		}
	}
	end = getTickCount();
	totalTime += end - start;
	cout << "reduce time: " << (end - start) / getTickFrequency() << endl;
	if (write)
		imwrite("./test2/boundingRect_m.jpg", imgR2_c);

	// scan
	start = getTickCount();
	Mat modifiedImg = pixelwiseScan(thresR2, roi1, write);
	Mat modifiedImg2 = pixelwiseScan(modifiedImg, roi2, write);
	Mat modifiedImg3 = pixelwiseScan(modifiedImg2, roi3, write);
	Mat modifiedImg4 = pixelwiseRefine(modifiedImg3, roi1, write);
	Mat modifiedImg5 = pixelwiseRefine(modifiedImg4, roi2, write);
	Mat modifiedImg6 = pixelwiseRefine(modifiedImg5, roi3, write);
	//Mat modifiedImg3 = pixelwiseScan(thresR2, Rect(0, 0, thresR2.cols, thresR2.rows), write);
	end = getTickCount();
	totalTime += end - start;
	cout << "scan time: " << (end - start) / getTickFrequency() << endl;

	// find and draw contour
	start = getTickCount();
	vector<vector<Point>> screenContours = findAndDrawContour(modifiedImg6, imgR2, "./test2/screen.jpg", write);
	end = getTickCount();
	totalTime += end - start;
	cout << "find time: " << (end - start) / getTickFrequency() << endl;

	// get point sets
	start = getTickCount();
	vector<Point> points1;
	vector<Point> points2;
	vector<Point> points3;
	for (auto contour : screenContours)
	{
		for (Point contourPoint : contour)
		{
			if (roi1_m.contains(contourPoint))
				points1.push_back(contourPoint);
			if (roi2_m.contains(contourPoint))
				points2.push_back(contourPoint);
			if (roi3_m.contains(contourPoint))
				points3.push_back(contourPoint);
		}
	}
	if (write)
		cout << "screen contour num: " << screenContours.size() << endl;
	// fit lines 
	Mat lineResult = imgR2_c;
	Vec4f lineParam1, lineParam2, lineParam3;
	fitLine(points1, lineParam1, DIST_L2, 0, 1e-2, 1e-2);
	fitLine(points2, lineParam2, DIST_L2, 0, 1e-2, 1e-2);
	fitLine(points3, lineParam3, DIST_L2, 0, 1e-2, 1e-2);
	//lineParam1[3] -= 710;  //test
	//lineParam2[2] -= 710;  //test
	//lineParam3[3] += 711;  //test
	//write = true;	//test
	if (write)
	{
		lineResult = drawLine(lineParam1, lineResult, 0, 5);
		lineResult = drawLine(lineParam2, lineResult, 1, 5);
		lineResult = drawLine(lineParam3, lineResult, 0, 5);
		imwrite("./test2/line.jpg", lineResult);
	}
	end = getTickCount();
	totalTime += end - start;
	cout << "fit time: " << (end - start) / getTickFrequency() << endl;

	// translation
	// threshold
	start = getTickCount();
	Mat thresOil;
	threshold(imgR2, thresOil, flag[1], 255, THRESH_TOZERO_INV);
	threshold(thresOil, thresOil, flag[0], 255, THRESH_TOZERO);
	if (write)
		imwrite("./test2/thresOil.jpg", thresOil);
	// calculate shift
	int colNum = 100;
	int rowStart = 1000;
	int rowEnd = 10000;
	int pixelIndex;
	int imgWidth = thresOil.cols;
	int* shiftCount = new int[colNum];
	int totalCount = 0;
	int shiftAmount;
	for (int i = 0; i < colNum; i++)
		shiftCount[i] = 0;
	for (int i = imgWidth - colNum; i < imgWidth; i++)
	{
		for (int j = rowStart; j < rowEnd; j++)
		{
			pixelIndex = j * imgWidth + i;
			if (thresOil.data[pixelIndex] > 0)
			{
				shiftCount[i - (imgWidth - colNum)]++;
			}
			if (thresOil.data[pixelIndex] == 0 && shiftCount[i - (imgWidth - colNum)] > 0)
			{
				totalCount += shiftCount[i - (imgWidth - colNum)];
				//cout << shiftCount[i - (imgWidth - colNum)] << endl;
				break;
			}
		}
	}
	delete[]shiftCount;
	shiftCount = nullptr;
	shiftAmount = totalCount / colNum;
	//draw lines
	lineParam1[3] -= shiftAmount;  
	lineParam2[2] -= shiftAmount;  
	lineParam3[3] += shiftAmount; 
	if (write)
	{
		lineResult = drawLine(lineParam1, lineResult, 0, 5);
		lineResult = drawLine(lineParam2, lineResult, 1, 5);
		lineResult = drawLine(lineParam3, lineResult, 0, 5);
		imwrite("./test2/oilLine.jpg", lineResult);
	}
	end = getTickCount();
	totalTime += end - start;
	cout << "oilLine time: " << (end - start) / getTickFrequency() << endl;
	cout << "total time: " << totalTime / getTickFrequency() << endl;
}

void rightSide()
{
	//Mat imgR1 = imread("./Origin2/0/1.jpg", IMREAD_GRAYSCALE);
	Mat imgR2 = imread("./Origin2/2/2.jpg", IMREAD_GRAYSCALE);
	Mat imgR2_c = imread("./Origin2/2/2.jpg", IMREAD_COLOR);

	bool write = false;
	bool showHist = false;

	int64 start, end;
	int64 totalTime = 0;

	// threshold
	start = getTickCount();
	Mat thresR2;
	drawHist(&imgR2, write, showHist);
	int bias = 10;
	threshold(imgR2, thresR2, flag[1] - bias, 255, THRESH_BINARY);
	//threshold(imgR2, thresR2, 0, 255, THRESH_OTSU);
	end = getTickCount();
	totalTime += end - start;
	cout << "thresh time: " << (end - start) / getTickFrequency() << endl;
	if (write)
		imwrite("./test2/threshTest.jpg", thresR2);
	
	// reduce ROI
	start = getTickCount();
	int roi1_width;
	int roi1_height = 1000;  //1000
	int roi2_width = 1000;
	int roi2_height;
	int roi3_width;
	int roi3_height = 1000;  //1000
	int modifyAmount = 200;
	vector<vector<Point>> contours;
	findContours(thresR2, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	// filter contour
	vector<Rect> areas;
	Rect roi1;
	Rect roi2;
	Rect roi3;
	Rect roi1_m;
	Rect roi2_m;
	Rect roi3_m;
	for (auto contour : contours)
	{
		auto box = boundingRect(contour);
		if (box.width < 2000 || box.height < 2000)
			continue;
		areas.push_back(box);
		// get ROI
		roi1_width = box.br().x - box.tl().x;
		roi2_height = box.br().y - box.tl().y;
		roi3_width = box.br().x - box.tl().x;
		//cout << box.tl() << endl;
		roi1 = Rect(box.tl().x, box.tl().y, roi1_width - roi2_width, roi1_height);
		roi2 = Rect(box.tl().x + roi1_width - roi2_width, box.tl().y, roi2_width, roi2_height);
		roi3 = Rect(box.br().x - roi3_width, box.br().y - roi3_height, roi3_width - roi2_width, roi3_height);
		roi1_m = Rect(box.tl().x + modifyAmount, box.tl().y, roi1_width - roi2_width - 2 * modifyAmount, roi1_height);
		roi2_m = Rect(box.tl().x + roi1_width - roi2_width, box.tl().y + roi1_height, roi2_width, roi2_height - 2 * roi3_height);
		roi3_m = Rect(box.br().x - roi3_width + modifyAmount, box.br().y - roi3_height, roi3_width - roi2_width - 2 * modifyAmount, roi3_height);
		if (write)
		{
			//rectangle(imgR2_c, box, Scalar(255, 0, 255), 10);
			rectangle(imgR2_c, roi1_m, Scalar(0, 0, 255), 10);
			rectangle(imgR2_c, roi2_m, Scalar(0, 255, 0), 10);
			rectangle(imgR2_c, roi3_m, Scalar(255, 0, 0), 10);
		}
	}
	end = getTickCount();
	totalTime += end - start;
	cout << "reduce time: " << (end - start) / getTickFrequency() << endl;
	if (write)
		imwrite("./test2/boundingRect_m.jpg", imgR2_c);
	
	// scan
	start = getTickCount();
	Mat modifiedImg = pixelwiseScan(thresR2, roi1, write);
	Mat modifiedImg2 = pixelwiseScan(modifiedImg, roi2, write);
	Mat modifiedImg3 = pixelwiseScan(modifiedImg2, roi3, write);
	Mat modifiedImg4 = pixelwiseRefine(modifiedImg3, roi1, write);
	Mat modifiedImg5 = pixelwiseRefine(modifiedImg4, roi2, write);
	Mat modifiedImg6 = pixelwiseRefine(modifiedImg5, roi3, write);
	end = getTickCount();
	totalTime += end - start;
	cout << "scan time: " << (end - start) / getTickFrequency() << endl;

	// find and draw contour
	start = getTickCount();
	vector<vector<Point>> screenContours = findAndDrawContour(modifiedImg6, imgR2, "./test2/screen.jpg", write);
	end = getTickCount();
	totalTime += end - start;
	cout << "find time: " << (end - start) / getTickFrequency() << endl;
	
	// get point sets
	start = getTickCount();
	vector<Point> points1;
	vector<Point> points2;
	vector<Point> points3;
	for (auto contour : screenContours)
	{
		for (Point contourPoint : contour)
		{
			if (roi1_m.contains(contourPoint))
				points1.push_back(contourPoint);
			if (roi2_m.contains(contourPoint))
				points2.push_back(contourPoint);
			if (roi3_m.contains(contourPoint))
				points3.push_back(contourPoint);
		}
	}
	if (write)
		cout << "screen contour num: " << screenContours.size() << endl;
	// fit lines 
	Mat lineResult = imgR2_c;
	Vec4f lineParam1, lineParam2, lineParam3;
	fitLine(points1, lineParam1, DIST_L2, 0, 1e-2, 1e-2);
	fitLine(points2, lineParam2, DIST_L2, 0, 1e-2, 1e-2);
	fitLine(points3, lineParam3, DIST_L2, 0, 1e-2, 1e-2);
	//lineParam1[3] -= 710;  //test
	//lineParam2[2] -= 710;  //test
	//lineParam3[3] += 711;  //test
	//write = true;	//test
	if (write)
	{
		lineResult = drawLine(lineParam1, lineResult, 0, 5);
		lineResult = drawLine(lineParam2, lineResult, 1, 5);
		lineResult = drawLine(lineParam3, lineResult, 0, 5);
		imwrite("./test2/line.jpg", lineResult);
	}
	end = getTickCount();
	totalTime += end - start;
	cout << "fit time: " << (end - start) / getTickFrequency() << endl;
	
	// translation
	// threshold
	start = getTickCount();
	Mat thresOil;
	threshold(imgR2, thresOil, flag[1], 255, THRESH_TOZERO_INV);
	threshold(thresOil, thresOil, flag[0], 255, THRESH_TOZERO);
	if (write)
		imwrite("./test2/thresOil.jpg", thresOil);
	// calculate shift
	int colNum = 100;
	int rowStart = 1000;
	int rowEnd = 10000;
	int pixelIndex;
	int imgWidth = thresOil.cols;
	int* shiftCount = new int[colNum];
	int totalCount = 0;
	int shiftAmount;
	for (int i = 0; i < colNum; i++)
		shiftCount[i] = 0;
	for (int i = 0; i < colNum; i++)
	{
		for (int j = rowStart; j < rowEnd; j++)
		{
			pixelIndex = j * imgWidth + i;
			if (thresOil.data[pixelIndex] > 0)
			{
				shiftCount[i]++;
			}
			if (thresOil.data[pixelIndex] == 0 && shiftCount[i] > 0)
			{
				totalCount += shiftCount[i];
				//cout << shiftCount[i] << endl;
				break;
			}
		}
	}
	delete[]shiftCount;
	shiftCount = nullptr;
	shiftAmount = totalCount / colNum;
	//cout << shiftAmount << endl;
	//draw lines
	lineParam1[3] -= shiftAmount;
	lineParam2[2] += shiftAmount;
	lineParam3[3] += shiftAmount;
	if (write)
	{
		lineResult = drawLine(lineParam1, lineResult, 0, 5);
		lineResult = drawLine(lineParam2, lineResult, 1, 5);
		lineResult = drawLine(lineParam3, lineResult, 0, 5);
		imwrite("./test2/oilLine.jpg", lineResult);
	}
	end = getTickCount();
	totalTime += end - start;
	cout << "oilLine time: " << (end - start) / getTickFrequency() << endl;
	cout << "total time: " << totalTime / getTickFrequency() << endl;
}

void middleSide()
{
	//Mat imgR1 = imread("./Origin2/0/1.jpg", IMREAD_GRAYSCALE);
	Mat imgR2 = imread("./Origin2/1/2.jpg", IMREAD_GRAYSCALE);
	Mat imgR2_c = imread("./Origin2/1/2.jpg", IMREAD_COLOR);

	bool write = false;
	bool showHist = false;

	int64 start, end;
	int64 totalTime = 0;

	// threshold
	start = getTickCount();
	Mat thresR2;
	drawHist(&imgR2, write, showHist);
	int bias = 10;
	threshold(imgR2, thresR2, flag[1] - bias, 255, THRESH_BINARY);
	//threshold(imgR2, thresR2, 0, 255, THRESH_OTSU);
	end = getTickCount();
	totalTime += end - start;
	cout << "thresh time: " << (end - start) / getTickFrequency() << endl;
	if (write)
		imwrite("./test2/threshTest.jpg", thresR2);
	
	// reduce ROI
	start = getTickCount();
	int roi1_width;
	int roi1_height = 1000;  //1000
	int roi3_width;
	int roi3_height = 1000;  //1000
	int modifyAmount = 200;
	vector<vector<Point>> contours;
	findContours(thresR2, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	// filter contour
	vector<Rect> areas;
	Rect roi1;
	Rect roi3;
	Rect roi1_m;
	Rect roi3_m;
	for (auto contour : contours)
	{
		auto box = boundingRect(contour);
		if (box.width < 2000 || box.height < 2000)
			continue;
		areas.push_back(box);
		// get ROI
		roi1_width = box.br().x - box.tl().x;
		roi3_width = box.br().x - box.tl().x;
		//cout << box.tl() << endl;
		roi1 = Rect(box.tl().x, box.tl().y, roi1_width, roi1_height);
		roi3 = Rect(box.br().x - roi3_width, box.br().y - roi3_height, roi3_width, roi3_height);
		roi1_m = Rect(box.tl().x + modifyAmount, box.tl().y, roi1_width - 2 * modifyAmount, roi1_height);
		roi3_m = Rect(box.br().x - roi3_width + modifyAmount, box.br().y - roi3_height, roi3_width - 2 * modifyAmount, roi3_height);
		if (write)
		{
			//rectangle(imgR2_c, box, Scalar(255, 0, 255), 10);
			rectangle(imgR2_c, roi1_m, Scalar(0, 0, 255), 10);
			rectangle(imgR2_c, roi3_m, Scalar(255, 0, 0), 10);
		}
	}
	end = getTickCount();
	totalTime += end - start;
	cout << "reduce time: " << (end - start) / getTickFrequency() << endl;
	if (write)
		imwrite("./test2/boundingRect_m.jpg", imgR2_c);
	
	// scan
	start = getTickCount();
	Mat modifiedImg = pixelwiseScan(thresR2, roi1, write);
	Mat modifiedImg3 = pixelwiseScan(modifiedImg, roi3, write);
	Mat modifiedImg4 = pixelwiseRefine(modifiedImg3, roi1, write);
	Mat modifiedImg6 = pixelwiseRefine(modifiedImg4, roi3, write);
	end = getTickCount();
	totalTime += end - start;
	cout << "scan time: " << (end - start) / getTickFrequency() << endl;

	// find and draw contour
	start = getTickCount();
	vector<vector<Point>> screenContours = findAndDrawContour(modifiedImg6, imgR2, "./test2/screen.jpg", write);
	end = getTickCount();
	totalTime += end - start;
	cout << "find time: " << (end - start) / getTickFrequency() << endl;
	
	// get point sets
	start = getTickCount();
	vector<Point> points1;
	vector<Point> points3;
	for (auto contour : screenContours)
	{
		for (Point contourPoint : contour)
		{
			if (roi1_m.contains(contourPoint))
				points1.push_back(contourPoint);
			if (roi3_m.contains(contourPoint))
				points3.push_back(contourPoint);
		}
	}
	if (write)
		cout << "screen contour num: " << screenContours.size() << endl;
	// fit lines 
	Mat lineResult = imgR2_c;
	Vec4f lineParam1, lineParam3;
	fitLine(points1, lineParam1, DIST_L2, 0, 1e-2, 1e-2);
	fitLine(points3, lineParam3, DIST_L2, 0, 1e-2, 1e-2);
	if (write)
	{
		lineResult = drawLine(lineParam1, lineResult, 0, 5);
		lineResult = drawLine(lineParam3, lineResult, 0, 5);
		imwrite("./test2/line.jpg", lineResult);
	}
	end = getTickCount();
	totalTime += end - start;
	cout << "fit time: " << (end - start) / getTickFrequency() << endl;
	
	// translation
	// threshold
	start = getTickCount();
	Mat thresOil;
	threshold(imgR2, thresOil, flag[1], 255, THRESH_TOZERO_INV);
	threshold(thresOil, thresOil, flag[0], 255, THRESH_TOZERO);
	if (write)
		imwrite("./test2/thresOil.jpg", thresOil);
	// calculate shift
	int colNum = 100;
	int rowStart = 1000;
	int rowEnd = 10000;
	int pixelIndex;
	int imgWidth = thresOil.cols;
	int* shiftCount = new int[colNum];
	int totalCount = 0;
	int shiftAmount;
	for (int i = 0; i < colNum; i++)
		shiftCount[i] = 0;
	for (int i = 0; i < colNum; i++)
	{
		for (int j = rowStart; j < rowEnd; j++)
		{
			pixelIndex = j * imgWidth + i;
			if (thresOil.data[pixelIndex] > 0)
			{
				shiftCount[i]++;
			}
			if (thresOil.data[pixelIndex] == 0 && shiftCount[i] > 0)
			{
				totalCount += shiftCount[i];
				//cout << shiftCount[i] << endl;
				break;
			}
		}
	}
	delete[]shiftCount;
	shiftCount = nullptr;
	shiftAmount = totalCount / colNum;
	//cout << shiftAmount << endl;
	//draw lines
	lineParam1[3] -= shiftAmount;
	lineParam3[3] += shiftAmount;
	if (write)
	{
		lineResult = drawLine(lineParam1, lineResult, 0, 5);
		lineResult = drawLine(lineParam3, lineResult, 0, 5);
		imwrite("./test2/oilLine.jpg", lineResult);
	}
	end = getTickCount();
	totalTime += end - start;
	cout << "oilLine time: " << (end - start) / getTickFrequency() << endl;
	cout << "total time: " << totalTime / getTickFrequency() << endl;
}

int main()
{
	middleSide();
	return 0;
}