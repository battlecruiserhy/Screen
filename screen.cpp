#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <fstream>
#include <iostream>

#include <omp.h>

using namespace cv;
using namespace std;

void getMidArea()
{
	Mat img = imread("window.jpg", 0);
	Mat res;
	Canny(img, res, 50, 150);
	imshow("window", res);
	waitKey(0);
	imwrite("mid.jpg", res);




	Mat sImg = imread("subRes.jpg", 0);
	Mat thresImg0, thresImg1;
	//threshold(sImg, thresImg0, 50, 255, THRESH_TOZERO);
	threshold(sImg, thresImg1, 10, 255, THRESH_OTSU);

	vector<vector<Point>> contours;
	vector<vector<Point>> finalContours;
	findContours(thresImg1, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

	vector<RotatedRect> areas;
	for (auto contour : contours)
	{
		auto box = minAreaRect(contour);
		if (box.size.width < 20 || box.size.height < 20)
			continue;
		areas.push_back(box);
		finalContours.push_back(contour);
	}
	cout << finalContours.size() << endl;

	for (int i = 0; i < finalContours.size(); i++)
	{
		vector<Point> tempCon = finalContours[i];
		cout << tempCon.size() << endl;
		for (int j = 0; j < tempCon.size(); j++)
		{
			//cout << "x: " << tempCon[j].x << " y: " << tempCon[j].y << endl;

		}
	}

	imshow("window", thresImg1);
	waitKey(0);

	//Mat thresColor;
	//cvtColor(thresImg1, thresColor, COLOR_GRAY2RGB);
	////Mat tempColor;
	//cvtColor(sImg, tempColor, COLOR_GRAY2RGB);
	//for (int i = 0; i < finalContours.size(); i++)
	//{
	//	drawContours(tempColor, finalContours, i, CV_RGB(255, 0, 0));
	//}
	//imshow("window", tempColor);
	//waitKey(0);
}

void getScreenArea()
{
	Mat r1 = imread("1.jpg", 0);
	Mat r2 = imread("2.jpg", 0);
	Mat subResult = r2 - r1;

	Mat thresResult;
	//threshold(subResult, thresResult, 40, 255, THRESH_BINARY);
	threshold(subResult, thresResult, 0, 255, THRESH_OTSU);
	//imwrite("tr.jpg",thresResult);
	vector<vector<Point>> contours;
	vector<vector<Point>> finalContours;
	findContours(thresResult, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	vector<RotatedRect> areas;
	for (auto contour : contours)
	{
		auto box = minAreaRect(contour);
		if (box.size.width < 2000 || box.size.height < 2000)
			continue;
		areas.push_back(box);
		finalContours.push_back(contour);
	}
	cout << finalContours.size() << endl;
	Mat thresColor;
	cvtColor(thresResult, thresColor, COLOR_GRAY2RGB);
	Mat tempColor;
	cvtColor(r2, tempColor, COLOR_GRAY2RGB);
	for (int i = 0; i < finalContours.size(); i++)
	{
		drawContours(tempColor, finalContours, i, CV_RGB(255, 0, 0));
	}
	imshow("window", tempColor);
	waitKey(0);
	imwrite("try_real_screen.jpg", tempColor);
}

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

void drawHist(Mat* img)
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
	cout <<"count num: "+to_string(count) << endl;
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
		if (contourTableY[i-1]>10 && contourTableY[i]<=2)
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
	delete[]contourTableY;
	delete[]contourTableX;
	contourTableY = nullptr;
	contourTableX = nullptr;

}

void doubleThreshold(Mat* img)
{
	flag[0] = 0;
	flag[1] = 0;
	Mat thresRes;
	drawHist(img);
	threshold(*img, thresRes, flag[1], 255, THRESH_TOZERO_INV);	
	//drawHist(&thresResult);
	threshold(thresRes, *img, flag[0], 255, THRESH_TOZERO);
}

void getOilArea()
{
	Mat r2r = imread("./Origin2/0/2.jpg", 0);
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
	//medianBlur(r1, r1, 7);
	//medianBlur(r2, r2, 7);
	Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
	Mat temp1, temp2;
	//erode(r1, temp1, kernel, Point(-1, -1), 5);
	//dilate(temp1, temp1, kernel, Point(-1, -1), 5);
	//erode(r2, temp2, kernel, Point(-1, -1), 5);
	//dilate(temp2, temp2, kernel, Point(-1, -1), 5);
	//imwrite("./result/temp1.jpg", temp1);
	temp1 = r1;
	temp2 = r2;

	Mat subRes = temp2 - temp1;
	imwrite("./result/subRes.jpg", subRes);
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
	imwrite("./result/thresResult2.jpg", thresResult2);
	//imshow("window", thresResult2);
	//imwrite("thresr1_thresResRes.jpg", thresResult2);
	//waitKey(0);
	//blur(thresResult2, thresResult2, Size(7, 7));
	
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
	//imwrite("./result/test.jpg", tempColor);


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
					if (f1x == f2x && f1y>replaceY)
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

void findAndDrawContour(Mat img, Mat src, string dst, int writeFlag)
{
	// find contour
	vector<vector<Point>> contours;
	vector<vector<Point>> finalContours;
	findContours(img, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
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
		drawContours(localTempColor, finalContours, i, CV_RGB(255, 0, 0));
	}
	if(writeFlag == 1)
		imwrite(dst, localTempColor);
}

void getScreenArea()
{
	double start, end;
	//namedWindow("window", cv::WINDOW_NORMAL);
	//resizeWindow("window", cv::Size(640, 720));	//640,720
	//combineImages();
	//getScreenArea();
	//getOilArea();


	//Mat img = imread("./test2/thresResult2.jpg", 0);
	Mat imgR1 = imread("./Origin2/0/1.jpg", IMREAD_GRAYSCALE);
	Mat imgR2 = imread("./Origin2/0/2.jpg", IMREAD_GRAYSCALE);
	start = getTickCount();


	Mat thresR;
	threshold(imgR1, thresR, 0, 255, THRESH_OTSU);


	//Mat thresRes = r1;
	//int height = thresRes.rows;
	//int width = thresRes.cols;
	//for (int i = 0; i < width; i++)
	//{
	//	for (int j = 0; j < height-1; j++)
	//	{
	//		int index = j * width + i;
	//		int postIndex = (j + 1) * width + i;


	Mat modifyImg = thresR.clone();
	int height = modifyImg.rows;
	int width = modifyImg.cols;
	int index, aboveIndex, frontIndex, backIndex, belowIndex;
	int score = 0;
	int fillColor = 0;
	int fillThreshold = 30;
	bool continuousWhite = false;
	int tempIndex;

	for (int i = 1; i < width - 1; i++)
	{
		score = 0;
		for (int j = 1; j < height - 1; j++)
		{
			index = j * width + i;
			if (modifyImg.data[index] > 0)
			{
				//aboveIndex = (j - 1) * width + i;
				belowIndex = (j + 1) * width + i;
				//frontIndex = j * width + (i - 1);
				//backIndex = j * width + (i + 1);

				score++;
				if (modifyImg.data[belowIndex] == 0)
				{
					if (score < fillThreshold)
					{
						while (score > 0)
						{
							tempIndex = (j - score + 1) * width + i;
							modifyImg.data[tempIndex] = 0;
							score--;
							//cout << score << endl;
						}
					}
					else
						score = 0;
				}

			}
		}
	}
	Mat sub = imgR2 - modifyImg;

	//Mat sub = imgR2 - imgR1;

	start = getTickCount();
	Mat thresR2;
	threshold(imgR2, thresR2, 80, 255, THRESH_BINARY);	//sub
	//threshold(sub, thresR2, 0, 255, THRESH_OTSU);
	//imwrite("./test2/threshTest.jpg", thresR2);
	end = getTickCount();
	cout << "thresh time: " << (end - start) / getTickFrequency() << endl;

	start = getTickCount();
	Mat modifyImg1 = thresR2.clone();
	height = modifyImg1.rows;
	width = modifyImg1.cols;
	score = 0;
	fillColor = 0;
	fillThreshold = 30;
#pragma omp parallel for
	for (int i = 1; i < width - 1; i++)
	{
		score = 0;
		for (int j = 1; j < height - 1; j++)
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
	end = getTickCount();
	cout << "modify1 time: " << (end - start) / getTickFrequency() << endl;

	start = getTickCount();
	Mat modifyImg2 = modifyImg1.clone();
	height = modifyImg2.rows;
	width = modifyImg2.cols;
	score = 0;
	fillColor = 0;
	fillThreshold = 5;
	for (int i = 1; i < height - 1; i++)
	{
		score = 0;
		for (int j = 1; j < width - 1; j++)
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
	end = getTickCount();
	cout << "modify2 time: " << (end - start) / getTickFrequency() << endl;

	start = getTickCount();
	findAndDrawContour(modifyImg2, imgR2, "./test2/screen.jpg", 1);


	//Mat newSub = sub - thresR2;
	//Mat thresR3;
	//threshold(newSub, thresR3, 0, 255, THRESH_OTSU);



	end = getTickCount();
	cout << "find time: " << (end - start) / getTickFrequency() << endl;
	//imwrite("./test2/modifyImg5.jpg", modifyImg);
	//imwrite("./test2/testImg.jpg", modifyImg2);
}

void test()
{
	double start, end;

	//getMidArea();


	Mat r2 = imread("./Origin2/0/2.jpg", 0);
	//vector<vector<Point>> contours;
	//vector<vector<Point>> finalContours;
	//threshold(r2, r2, 10, 255, THRESH_BINARY);
	//findContours(r2, contours, RETR_LIST, CHAIN_APPROX_NONE);
	//vector<RotatedRect> areas;
	//for (auto contour : contours)
	//{
	//	auto box = minAreaRect(contour);
	//	if (box.size.width < 2000 || box.size.height < 500)
	//		continue;
	//	areas.push_back(box);
	//	finalContours.push_back(contour);
	//}
	//cout << finalContours.size() << endl;
	//Mat thresColor;
	//cvtColor(r2, tempColor, COLOR_GRAY2RGB);
	//for (int i = 0; i < finalContours.size(); i++)
	//{
	//	drawContours(tempColor, finalContours, i, CV_RGB(255, 0, 0));
	//}
	//imwrite("test4.jpg", tempColor);



	Mat r1 = imread("./Origin2/0/1.jpg", 0);

	//// preprocess
	//medianBlur(r1, r1, 7);
	//medianBlur(r2, r2, 7);
	//Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
	Mat temp1, temp2;
	//erode(r1, temp1, kernel, Point(-1, -1), 5);
	//dilate(temp1, temp1, kernel, Point(-1, -1), 5);
	//erode(r2, temp2, kernel, Point(-1, -1), 5);
	//dilate(temp2, temp2, kernel, Point(-1, -1), 5);

	temp1 = r1;
	temp2 = r2;

	Mat dst;
	dst = temp1 - temp2;
	imwrite("./test/subRes.jpg", dst);
	Mat tImg1, tImg2;

	//drawHist(&dst);
	//threshold(dst, tImg1, 200, 255, THRESH_TOZERO_INV);		//200
	threshold(dst, tImg2, 200, 255, THRESH_TOZERO_INV);
	//drawHist(&thresResult);
	//threshold(tImg1, tImg2, 0, 255, THRESH_TOZERO);		//170
	//imshow("window", tImg2);
	//waitKey(0);
	Mat res;
	res = tImg2;

	//imshow("window", res);
	imwrite("./test/thresSubRes.jpg", res);
	//waitKey(0);


	temp1 = temp1 - 2 * res;  //3
	//imshow("window", temp1);
	imwrite("test1.jpg", temp1);
	//waitKey(0);


	Mat subRes;
	subRes = temp2 - temp1;
	//imshow("window", subRes);
	imwrite("test2.jpg", subRes);
	//waitKey(0);



	//// final process
	////finalProcess(subRes, r2);



	//destroyAllWindows();

	//




	//Mat r1 = imread("./Origin2/0/1.jpg", 0);

	//start = getTickCount();
	//Mat thresRes = r1;
	//int height = thresRes.rows;
	//int width = thresRes.cols;
	//for (int i = 0; i < width; i++)
	//{
	//	for (int j = 0; j < height-1; j++)
	//	{
	//		int index = j * width + i;
	//		int postIndex = (j + 1) * width + i;
	//		if ((int)thresRes.data[index]!=0 && (int)thresRes.data[index]< (int)thresRes.data[postIndex] && (int)thresRes.data[postIndex]>150)
	//		{
	//			thresRes.data[postIndex] = thresRes.data[index];
	//		}
	//	}
	//}
	//end = getTickCount();
	//cout << "total time: " << (end - start) / getTickFrequency() << endl;
	//imwrite("./test/linePorcessRes.jpg", thresRes);
}

void imgR1()
{
	Mat r2 = imread("./Origin2/0/2.jpg", 0);
	Mat r1 = imread("./Origin2/0/1.jpg", 0);
	//start = getTickCount();

	Mat dst;
	threshold(r2, dst, 5, 255, THRESH_BINARY_INV);	//	threshold:10(TODO)
	Mat sub;
	sub = r1 - dst;
	//threshold(sub, sub, 200, 255, THRESH_TOZERO_INV);
	//imwrite("./test/sub0.jpg", sub);
	// fill the write(threshold) point
	Mat modifyImg = sub;
	int height = modifyImg.rows;
	int width = modifyImg.cols;
	int index, aboveIndex, frontIndex, backIndex, belowIndex;
	bool modifyFlag = false;
	bool* whiteFlags = new bool[(long)height * (long)width];
	int value = 50;
	int whiteThreshold = 200;
	for (int i = 0; i < height * width; i++)
		whiteFlags[i] = false;
	for (int i = 1; i < height - 1; i++)
	{
		for (int j = 1; j < width - 1; j++)
		{
			index = i * width + j;
			if (modifyImg.data[index] > whiteThreshold)	// threshold:200(TODO)
			{
				aboveIndex = (i - 1) * width + j;
				//belowIndex = (i + 1) * width + j;
				frontIndex = i * width + (j - 1);
				//backIndex = i * width + (j + 1);
				modifyFlag = false;
				if ((modifyImg.data[aboveIndex] == 0 || modifyImg.data[frontIndex] == 0) || (whiteFlags[aboveIndex] || whiteFlags[frontIndex]))
				{
					modifyFlag = true;
					whiteFlags[index] = true;
					//modifyImg.data[index] = 0;
				}
				if (modifyImg.data[aboveIndex] > 0 && modifyImg.data[aboveIndex] < whiteThreshold && modifyFlag == false)
				{
					//value = (int)modifyImg.data[aboveIndex];
					//cout << (int)modifyImg.data[aboveIndex]<< " "<< value << endl;
					modifyImg.data[index] = value;
					//cout << (int)modifyImg.data[index] << endl;
					modifyImg.data[aboveIndex] = value;
					//modifyImg.data[backIndex] = value;
				}
				if (modifyImg.data[frontIndex] > 0 && modifyImg.data[frontIndex] < whiteThreshold && modifyFlag == false)
				{
					//value = (int)modifyImg.data[frontIndex];
					modifyImg.data[index] = value;
					modifyImg.data[frontIndex] = value;
					//modifyImg.data[backIndex] = value;
				}
			}
		}
	}
	delete[]whiteFlags;
	whiteFlags = nullptr;
}

void tryR1()
{
	double start, end;
	Mat r2 = imread("./Origin2/0/2.jpg", 0);
	Mat r1 = imread("./Origin2/0/1.jpg", 0);
	start = getTickCount();

	Mat dst;
	threshold(r2, dst, 10, 255, THRESH_BINARY_INV);	//	threshold:10(TODO)
	Mat sub;
	sub = r1 - dst;
	//threshold(sub, sub, 200, 255, THRESH_TOZERO_INV);
	//imwrite("./test/sub0.jpg", sub);
	//drawHist(&sub);

	// fill the write(threshold) point
	Mat modifyImg = r1;
	int height = modifyImg.rows;
	int width = modifyImg.cols;
	int index, aboveIndex, frontIndex, backIndex, belowIndex;
	bool modifyFlag = false;
	bool* whiteFlags = new bool[(long)height * (long)width];
	int value = 60;
	int whiteThreshold = 100;
	for (int i = 0; i < height * width; i++)
		whiteFlags[i] = false;
	for (int i = 1; i < height - 1; i++)
	{
		for (int j = width - 1; j > 1; j--)
		{
			index = i * width + j;
			if (modifyImg.data[index] > whiteThreshold)	// threshold:200(TODO)
			{
				aboveIndex = (i - 1) * width + j;
				belowIndex = (i + 1) * width + j;
				frontIndex = i * width + (j - 1);
				backIndex = i * width + (j + 1);
				modifyFlag = false;
				//if ( (modifyImg.data[aboveIndex] == 0 || modifyImg.data[frontIndex] == 0) || (whiteFlags[aboveIndex] || whiteFlags[frontIndex]))
				//{
				//	modifyFlag = true;
				//	whiteFlags[index] = true;
				//	//modifyImg.data[index] = 0;
				//}
				//if (modifyImg.data[aboveIndex] > 0 && modifyImg.data[aboveIndex] < whiteThreshold && modifyFlag == false)
				//{
				//	//value = (int)modifyImg.data[aboveIndex];
				//	//cout << (int)modifyImg.data[aboveIndex]<< " "<< value << endl;
				//	modifyImg.data[index] = value;
				//	//cout << (int)modifyImg.data[index] << endl;
				//	modifyImg.data[aboveIndex] = value;
				//	//modifyImg.data[backIndex] = value;
				//}
				//if (modifyImg.data[frontIndex] > 0 && modifyImg.data[frontIndex] < whiteThreshold && modifyFlag == false)
				//{
				//	//value = (int)modifyImg.data[frontIndex];
				//	modifyImg.data[index] = value;
				//	modifyImg.data[frontIndex] = value;
				//	//modifyImg.data[backIndex] = value;
				//}
			}
			if (modifyImg.data[index] >= whiteThreshold && (modifyImg.data[aboveIndex] > 0 && modifyImg.data[aboveIndex] < whiteThreshold) && (modifyImg.data[backIndex] > 0 && modifyImg.data[backIndex] < whiteThreshold))
			{
				modifyImg.data[index] = value;
			}
		}
	}
	delete[]whiteFlags;
	whiteFlags = nullptr;

	//Mat test = imread("./test/linePorcessRes.jpg", 0);
	Mat sub2;
	//doubleThreshold(&sub);




	end = getTickCount();
	cout << "time: " << (end - start) / getTickFrequency() << endl;
	imwrite("./test/modifyImg.jpg", modifyImg);
	//imwrite("./test/sub1.jpg", sub2);
}

void tryThres()
{
	double start, end;
	Mat img = imread("./test2/thresResult2.jpg", 0);
	Mat imgR1 = imread("./Origin2/0/1.jpg", 0);
	Mat imgR2 = imread("./Origin2/0/2.jpg", 0);
	start = getTickCount();

	Mat modifyImg = img.clone();
	int height = modifyImg.rows;
	int width = modifyImg.cols;
	int index, aboveIndex, frontIndex, backIndex, belowIndex;
	int score;
	int fillColor = 255;
	for (int i = 1; i < height - 1; i++)
	{
		for (int j = 1; j < width - 1; j++)
		{
			score = 0;
			index = i * width + j;
			if (modifyImg.data[index] == 0)
			{
				aboveIndex = (i - 1) * width + j;
				belowIndex = (i + 1) * width + j;
				frontIndex = i * width + (j - 1);
				backIndex = i * width + (j + 1);
				if (modifyImg.data[aboveIndex] > 0)
					score++;
				if (modifyImg.data[belowIndex] > 0)
					score++;
				if (modifyImg.data[frontIndex] > 0)
					score++;
				if (modifyImg.data[backIndex] > 0)
					score++;
				// fill color
				if (score >= 2 && (imgR1.data[index] > 20) && (imgR2.data[index] > 5 && imgR2.data[index] < 90))
					modifyImg.data[index] = fillColor;

				if (score >= 2 && (imgR2.data[index] > 100) && (imgR1.data[index] > 5 && imgR1.data[index] < 90))
					modifyImg.data[index] = fillColor;

			}
		}
	}
	//Mat sub = imgR2 - modifyImg;
	end = getTickCount();
	cout << "time: " << (end - start) / getTickFrequency() << endl;
	imwrite("./test2/modifyImg4.jpg", modifyImg);
}

int main()
{
	double start, end;
	//namedWindow("window", cv::WINDOW_NORMAL);
	//resizeWindow("window", cv::Size(640, 720));	//640,720
	//combineImages();
	//getScreenArea();
	//getOilArea();


	//Mat img = imread("./test2/thresResult2.jpg", 0);
	Mat imgR1 = imread("./Origin2/0/1.jpg", IMREAD_GRAYSCALE);
	Mat imgR2 = imread("./Origin2/0/2.jpg", IMREAD_GRAYSCALE);
	start = getTickCount();


	Mat thresR;
	threshold(imgR1, thresR, 0, 255, THRESH_OTSU);


	//Mat thresRes = r1;
	//int height = thresRes.rows;
	//int width = thresRes.cols;
	//for (int i = 0; i < width; i++)
	//{
	//	for (int j = 0; j < height-1; j++)
	//	{
	//		int index = j * width + i;
	//		int postIndex = (j + 1) * width + i;


	Mat modifyImg = thresR.clone();
	int height = modifyImg.rows;
	int width = modifyImg.cols;
	int index, aboveIndex, frontIndex, backIndex, belowIndex;
	int score = 0;
	int fillColor = 0;
	int fillThreshold = 30;
	bool continuousWhite = false;
	int tempIndex;

	for (int i = 1; i < width - 1; i++)
	{
		score = 0;
		for (int j = 1; j < height - 1; j++)
		{
			index = j * width + i;
			if (modifyImg.data[index] > 0)	
			{
				//aboveIndex = (j - 1) * width + i;
				belowIndex = (j + 1) * width + i;
				//frontIndex = j * width + (i - 1);
				//backIndex = j * width + (i + 1);

				score++;
				if (modifyImg.data[belowIndex] == 0)
				{
					if (score < fillThreshold)
					{
						while (score > 0)
						{
							tempIndex = (j - score + 1) * width + i;
							modifyImg.data[tempIndex] = 0;
							score--;
							//cout << score << endl;
						}
					}
					else
						score = 0;
				}

			}
		}
	}
	Mat sub = imgR2 - modifyImg;

	//Mat sub = imgR2 - imgR1;

	start = getTickCount();
	Mat thresR2;
	threshold(imgR2, thresR2, 80, 255, THRESH_BINARY);	//sub
	//threshold(sub, thresR2, 0, 255, THRESH_OTSU);
	//imwrite("./test2/threshTest.jpg", thresR2);
	end = getTickCount();
	cout << "thresh time: " << (end - start) / getTickFrequency() << endl;

	start = getTickCount();
	Mat modifyImg1 = thresR2.clone();
	height = modifyImg1.rows;
	width = modifyImg1.cols;
	score = 0;
	fillColor = 0;
	fillThreshold = 30;
#pragma omp parallel for
	for (int i = 1; i < width - 1; i++)
	{
		score = 0;
		for (int j = 1; j < height - 1; j++)
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
	end = getTickCount();
	cout << "modify1 time: " << (end - start) / getTickFrequency() << endl;

	start = getTickCount();
	Mat modifyImg2 = modifyImg1.clone();
	height = modifyImg2.rows;
	width = modifyImg2.cols;
	score = 0;
	fillColor = 0;
	fillThreshold = 5;
	for (int i = 1; i < height - 1; i++)
	{
		score = 0;
		for (int j = 1; j < width - 1; j++)
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
							modifyImg2.data[index-score+1] = 0;
							score--;
						}
					}
					else
						score = 0;
				}

			}
		}
	}
	end = getTickCount();
	cout << "modify2 time: " << (end - start) / getTickFrequency() << endl;

	start = getTickCount();
	findAndDrawContour(modifyImg2, imgR2, "./test2/screen.jpg", 1);


	//Mat newSub = sub - thresR2;
	//Mat thresR3;
	//threshold(newSub, thresR3, 0, 255, THRESH_OTSU);
	


	end = getTickCount();
	cout << "find time: " << (end - start) / getTickFrequency() << endl;
	//imwrite("./test2/modifyImg5.jpg", modifyImg);
	//imwrite("./test2/testImg.jpg", modifyImg2);

	return 0;
}