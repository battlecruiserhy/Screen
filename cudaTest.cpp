#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafilters.hpp>
using namespace std;
using namespace cv;
using namespace cv::cuda;

int main()
{


    Mat src_map = imread("./Origin2/0/2.jpg", 0);


    cuda::GpuMat GPU_src;
    GPU_src.upload(src_map);

    cuda::GpuMat GPU_open;
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5), Point(-1, -1));
    //Mat kernel = (cv::Mat_<float>(3, 3) << -1.0f, 0.0f, 1.0f, -2.0f, 0.0f, 2.0f, -1.0f, 0.0f, 1.0f);
    Ptr<Filter> open_Filter = cuda::createMorphologyFilter(MORPH_DILATE, CV_8UC1, kernel);
    open_Filter->apply(GPU_src, GPU_open);

    Mat open_map;
    GPU_open.download(open_map);
    //GPU_src.download(open_map);

    imwrite("done.jpg", open_map);

    return 0;
}