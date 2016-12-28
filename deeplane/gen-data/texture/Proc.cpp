#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <limits>
#include <time.h> 
#include <math.h>
using namespace std;
using namespace cv;

int main(int argc, char** argv )
{
    Mat texture = imread("wooden1.jpg", CV_LOAD_IMAGE_UNCHANGED);
    Rect myROI(0, 0, 2048, 2048);
    Mat wooden = texture(myROI);
    repeat(wooden, 2, 2, wooden);

    namedWindow("wooden", WINDOW_NORMAL);
    imshow("wooden", wooden);
    waitKey(0);
    destroyWindow("wooden");
}

// void genImage(int mid, double slope) {
//     Mat out;
//     int posx = rand() % (carpetX - 272);
//     int posy = rand() % (carpetY - 18);

//     Rect myROI(posx, posy, 272, 18);
//     out = carpet(myROI).clone();

//     Point2f top(mid + tan(slope * M_PI / 180) * 19, -10);
//     Point2f bottom(mid - tan(slope * M_PI / 180) * 18, 28);
//     int lthickness = rand() % 10 + 15;
//     line(out, top, bottom, Scalar(255, 255, 255), lthickness);

//     GaussianBlur(out, out, Size(5,5), 10);
//     cvtColor(out, out, CV_BGR2GRAY);

//     ostringstream ssDirName;
//     ssDirName << ((mid > 0 && mid <= 272) ? mid : 0);

//     string dirName = ssDirName.str();

//     ostringstream ss;
//     ss << "gen_image/" << dirName << "/" << g_count << ".jpg";
//     imwrite(ss.str(), out);
// }