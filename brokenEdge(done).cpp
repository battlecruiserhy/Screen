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

// tool functions
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

Point getCrossPoint(Vec4f lineParam1_1, Vec4f lineParam2_1)
{
	// calculate cross point
	float lk1 = lineParam1_1[1] / lineParam1_1[0];
	float lb1 = lineParam1_1[3] - lk1 * lineParam1_1[2];
	float lk2 = lineParam2_1[0] / lineParam2_1[1];
	float lb2 = lineParam2_1[2] - lk2 * lineParam2_1[3];
	float cx, cy;
	cy = (lk1 * lb2 + lb1) / (1 - lk1 * lk2);
	cx = lk2 * cy + lb2;
	Point crossPoint = Point2f(cx, cy);

	return crossPoint;
}

int findMostNum(vector<int> aCountSet)
{
	int maxFlag = 0;
	int max = 0;
	int mostNum = 0;
	int count[30];
	for (int i = 0; i < 30; i++)
		count[i] = 0;
	for (uint64 i = 0; i < aCountSet.size(); i++)
		count[aCountSet[i]]++;
	for (int i = 0; i < 30; i++)
	{
		if (count[i] > max)
		{
			max = count[i];
			maxFlag = i;
		}
	}
	mostNum = maxFlag;
	return mostNum;
}

// core functions
vector<Vec4f> findEgdesAndFitLines(Mat img, Rect box, Rect roi_line1, Rect roi_line2, Rect roi_line3)
{
	// find edges
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
		
		px = x;
		py = y;
	}

	// fit lines
	vector<Point> pointLines1;
	vector<Point> pointLines2;
	vector<Point> pointLines3;
	for (int i = 0; i < LinePoints1.size(); i++)
	{
		if (roi_line1.contains(LinePoints1[i]))
			pointLines1.push_back(LinePoints1[i]);
	}
	for (int i = 0; i < linePoints.size(); i++)
	{
		if (roi_line2.contains(linePoints[i]))
			pointLines2.push_back(linePoints[i]);
	}
	for (int i = 0; i < LinePoints3.size(); i++)
	{
		if (roi_line3.contains(LinePoints3[i]))
			pointLines3.push_back(LinePoints3[i]);
	}
	Vec4f lineParam1, lineParam2, lineParam3;
	fitLine(pointLines1, lineParam1, DIST_L2, 0, 1e-2, 1e-2);
	fitLine(pointLines2, lineParam2, DIST_L2, 0, 1e-2, 1e-2);
	fitLine(pointLines3, lineParam3, DIST_L2, 0, 1e-2, 1e-2);

	// store the result
	vector<Vec4f> resultLineParams;
	resultLineParams.push_back(lineParam1);
	resultLineParams.push_back(lineParam2);
	resultLineParams.push_back(lineParam3);

	return resultLineParams;
}

vector<Vec4f> findEdgesAfterRotate(Mat rotateImg, bool write)
{
	// get three edge after rotate
	Mat rotateImg_c;
	cvtColor(rotateImg, rotateImg_c, COLOR_GRAY2BGR);
		// line1
	int rIndex = 0;
	int rX = 0, rY = 0, rX1 = 0, rY1 = 0;
	for (int i = 0; i < rotateImg.cols; i++)
	{
		rIndex = 1 * rotateImg.cols + i;
		if (rotateImg.data[rIndex] > 200)
		{
			rX = i;
			break;
		}
	}
	for (int i = 1; i < rotateImg.rows; i++)
	{
		rIndex = i * rotateImg.cols + rX;
		if (rotateImg.data[rIndex] < 150)
		{
			rY = i;
			break;
		}
	}
	Vec4f rLineParam1 = Vec4f(1, 0, (float)rX, (float)rY);
	// line2
	for (int i = 0; i < rotateImg.rows; i++)
	{
		rIndex = i * rotateImg.cols + rotateImg.cols - 1;
		if (rotateImg.data[rIndex] > 200)
		{
			rY = i;
			break;
		}
	}
	for (int i = rotateImg.cols - 1; i > 0; i--)
	{
		rIndex = rY * rotateImg.cols + i;
		if (rotateImg.data[rIndex] < 150)
		{
			rX = i;
			break;
		}
	}
	Vec4f rLineParam2 = Vec4f(0, 1, (float)rX, (float)rY);
	// line3
	for (int i = rotateImg.cols; i > 0; i--)
	{
		rIndex = (rotateImg.rows - 1) * rotateImg.cols + i;
		if (rotateImg.data[rIndex] > 200)
		{
			rX = i;
			break;
		}
	}
	for (int i = rotateImg.rows - 1; i > 0; i--)
	{
		rIndex = i * rotateImg.cols + rX;
		if (rotateImg.data[rIndex] < 150)
		{
			rY = i;
			break;
		}
	}
	rX1 = rX - 1000;
	for (int i = rotateImg.rows - 1; i > 0; i--)
	{
		rIndex = i * rotateImg.cols + rX1;
		if (rotateImg.data[rIndex] < 150)
		{
			rY1 = i;
			break;
		}
	}
	Vec4f rLineParam3;
	vector<Point> rLine3Points;
	rLine3Points.push_back(Point(rX, rY));
	rLine3Points.push_back(Point(rX1, rY1));
	fitLine(rLine3Points, rLineParam3, DIST_L2, 0, 1e-2, 1e-2);
	// test
	if (write)
	{
		drawLine(rLineParam1, rotateImg_c, 0, 1);
		drawLine(rLineParam2, rotateImg_c, 1, 1);
		drawLine(rLineParam3, rotateImg_c, 0, 1);
		imwrite("./test/rotate.jpg", rotateImg_c);
	}
	// store the result
	vector<Vec4f> resultLineParams;
	resultLineParams.push_back(rLineParam1);
	resultLineParams.push_back(rLineParam2);
	resultLineParams.push_back(rLineParam3);

	return resultLineParams;
}

vector<Point> getOriginalArc(Mat rotateImg_c, Mat rotateImg, Vec4f rLineParam2, Vec4f rLineParam3, bool write)
{
	// get roi (run once)
	Point rCrossPoint1 = getCrossPoint(rLineParam3, rLineParam2);
	int roi_width = 300;
	int roi_height = 300;
	Rect roi = Rect(rCrossPoint1.x - roi_width, rCrossPoint1.y - roi_height, roi_width, roi_height);
	if (write)
		rectangle(rotateImg_c, roi, Scalar(255, 255, 0), 2);

	// get the original arc (run once)
	Mat roiMat_t = Mat(rotateImg, roi);
	Mat roiMat = roiMat_t.clone();
	Mat roiMat_c;
	cvtColor(roiMat, roiMat_c, COLOR_GRAY2BGR);
	vector<Point> originalArcPoints;
	int rIndex = 0;
	for (int i = 1; i <= roiMat.rows - 1; i++)
	{
		for (int j = roiMat.cols - 5; j > 0; j--)
		{
			rIndex = i * roiMat.rows + j;
			if (roiMat.data[rIndex] < 200 && j > 0)
			{
				originalArcPoints.push_back(Point(j, i));
				break;
			}
		}
	}

	// record
	fstream  arcFile;
	arcFile.open("./test/originalArcPoints.txt", ios::app | ios::out | ios::in);
	for (uint64 i = 0; i < originalArcPoints.size(); i++)
	{
		arcFile << originalArcPoints[i].x << " " << originalArcPoints[i].y << endl;
	}
	arcFile.close();

	// test
	if (write)
	{
		for (uint i = 0; i < originalArcPoints.size(); i++)
		{
			if (i > 0)
				line(roiMat_c, originalArcPoints[i - 1], originalArcPoints[i], Scalar(0, 255, 255), 1);
		}
		imwrite("./test/originalArc.jpg", roiMat_c);
	}

	return originalArcPoints;
}

vector<vector<Point>> findBrokenEdge(Mat rotateImg_c, vector<Point> referenceArcPoints, bool write)
{
	// find brokenEgde
	Mat shiftImg;
	cvtColor(rotateImg_c, shiftImg, COLOR_RGB2GRAY);
	int aIndex = 0, aLoopIndex = 0;
	int aX = 0, aY = 0;
	int aPX = 0, aPY = 0;
	vector<int> aCountSet;
	int aCount;
	vector<Point> aPoints;
	vector<Point> breakEdgePoints;
	vector<vector<Point>> brokenEdgeSet;
	int startX = referenceArcPoints[referenceArcPoints.size() - 1].x;
	int endX = referenceArcPoints[0].x;
	for (int i = startX; i < endX; i++)
	{
		for (int j = 0; j < shiftImg.rows; j++)
		{
			aIndex = j * shiftImg.cols + i;
			if (shiftImg.data[aIndex] > 50 && shiftImg.data[aIndex] < 100)    // problem: can't find all the points due to only the first point can be found -> solution: two "for" loop, when aY <  aPY, shift to the second loop
			{
				// calculate distance
				aX = i;
				aY = j;
				aLoopIndex = aY * shiftImg.cols + aX;
				aCount = 0;
				while (shiftImg.data[aLoopIndex] > 50)
				{
					aCount++;
					aX--;
					aY++;
					aLoopIndex = aY * shiftImg.cols + aX;
				}
				// test
				if (write)
				{
					fstream  afile;
					afile.open("./test/record.txt", ios::app | ios::out | ios::in);
					afile << aCount << endl;
					afile.close();
				}
				// record brokenEdge line
				aCountSet.push_back(aCount);
				aPoints.push_back(Point(aX, aY));
				// loop end
				break;
			}
		}
	}
	// find the abnormal points
	int mostNum = findMostNum(aCountSet);
	int threshold = 2;
	bool begin = false;
	for (uint64 i = 0; i < aCountSet.size(); i++)
	{
		if (aCountSet[i] > mostNum + threshold) // begin
		{
			begin = true;
			breakEdgePoints.push_back(aPoints[i]);
		}
		if (begin == true && (aCountSet[i] <= mostNum + threshold))  // end
		{
			brokenEdgeSet.push_back(breakEdgePoints);
			breakEdgePoints.clear();
			begin = false;
		}
	}
	return brokenEdgeSet;
}

// main function
Mat brokenEdge(Mat img, Mat img_c, bool write, bool getArc)
{
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
	// roi for three main lines
	Rect roi_line1 = Rect(box.tl().x + box.width / 4, box.tl().y, box.width / 2, 500);
	Rect roi_line2 = Rect(box.br().x - 500, box.tl().y + box.height / 4, 500, box.height / 2);
	Rect roi_line3 = Rect(box.tl().x + box.width / 4, box.br().y - 500, box.width / 2, 500);

	//  find egdes and fit lines
	vector<Vec4f> resultLineParams = findEgdesAndFitLines(img, box, roi_line1, roi_line2, roi_line3);
	Vec4f lineParam1 = resultLineParams[0];
	Vec4f lineParam2 = resultLineParams[1];
	Vec4f lineParam3 = resultLineParams[2];

	// rotate the img
	float tan = lineParam1[1] / lineParam1[0];
	double angle = atan(tan) * 180.0 / 3.14;
	Mat rotateMat = getRotationMatrix2D(Point(img_c.cols / 2, img_c.rows / 2), angle, 1);
	Mat rotateImg;
	warpAffine(img, rotateImg, rotateMat, img.size());
	Mat rotateImg_c;
	cvtColor(rotateImg, rotateImg_c, COLOR_GRAY2BGR);
	if (write)
		imwrite("./test/rotate.jpg", rotateImg);

	// get three edge after rotate
	vector<Vec4f> resultLineParamsAfterRotate = findEdgesAfterRotate(rotateImg, write);
	Vec4f rLineParam1 = resultLineParamsAfterRotate[0];
	Vec4f rLineParam2 = resultLineParamsAfterRotate[1];
	Vec4f rLineParam3 = resultLineParamsAfterRotate[2];

	// get the original arc
	uint64 originalArcPointsNum = 0;
	vector<Point> originalArcPoints;
	if (getArc)  // find it (only run once for one type of product)
	{
		originalArcPoints = getOriginalArc(rotateImg_c, rotateImg, rLineParam2, rLineParam3, write);
		originalArcPointsNum = originalArcPoints.size();
	}
	else // get the arc points from the file
	{
		fstream arcFile;
		arcFile.open("./test/originalArcPoints.txt", ios::app | ios::out | ios::in);
		char bufferX[256];
		char bufferY[256];
		Point tempPoint;
		while (!arcFile.eof())
		{
			arcFile >> bufferX;
			arcFile >> bufferY;
			tempPoint = Point(atoi(bufferX),atoi(bufferY));
			if(tempPoint.x > 0)
				originalArcPoints.push_back(tempPoint);
		}
		arcFile.close();
		originalArcPointsNum = originalArcPoints.size();
	}

	// set the reference arc
	Mat resultImg = rotateImg_c.clone();
	Point rCrossPoint2 = getCrossPoint(rLineParam1, rLineParam2);
	rCrossPoint2.x += 10;
	rCrossPoint2.y -= 10;
	int deltaX = 0, deltaY = 0, tX = 0, tY = 0;
	vector<Point> referenceArcPoints;
	for (uint i = 0; i < originalArcPointsNum; i++)
	{
		deltaX = 300 - originalArcPoints[i].x;
		deltaY = 300 - originalArcPoints[i].y;
		tX = rCrossPoint2.x - deltaX;
		tY = rCrossPoint2.y + deltaY;
		referenceArcPoints.push_back(Point(tX, tY));
	}
	for (uint64 i = 0; i < referenceArcPoints.size(); i++)
	{
		if (i > 0)
			line(rotateImg_c, referenceArcPoints[i - 1], referenceArcPoints[i], Scalar(255, 0, 0), 1);
	}
	if (write)
		imwrite("./test/setReference.jpg", rotateImg_c);

	// find brokenEgde
	vector<vector<Point>> breakEdgeSet = findBrokenEdge(rotateImg_c, referenceArcPoints, write);

	// draw break edge box
	vector<Point> breakEdgePoints;
	if (breakEdgeSet.size() == 0)
		return resultImg;
	cout << "num: " << breakEdgeSet.size() << endl;
	for (uint64 i = 0; i < breakEdgeSet.size(); i++)
	{
		breakEdgePoints = breakEdgeSet[i];
		int endPointNum = (int)breakEdgePoints.size() - 1;
		Rect breakEdgeRect = Rect(breakEdgePoints[0], breakEdgePoints[endPointNum]);
		if (!getArc)
			rectangle(resultImg, breakEdgeRect, Scalar(255, 255, 0), 2);
	}

	return resultImg;
}


int main()
{
	// TODO: exam multi-brokenEdge/none-brokenEgde

	bool write = false;
	bool getArc = false;

	int64 start, end;
	Mat img = imread("./Origin/0/463990d5-7b7e-4e0b-8979-5f86564c0b63_camera0-type5.bmp", IMREAD_GRAYSCALE);
	Mat img_c = imread("./Origin/0/463990d5-7b7e-4e0b-8979-5f86564c0b63_camera0-type5.bmp", IMREAD_COLOR);

	start = getTickCount();

	// main function
	Mat resultImg = brokenEdge(img, img_c, write, getArc);

	end = getTickCount();
	cout << "time: " << (end - start) / getTickFrequency() << endl;

	if (!getArc)
	{
		imwrite("./test/resultImg.jpg", resultImg);
		imwrite("./resultImg.jpg", resultImg);
	}

	return 0;
}