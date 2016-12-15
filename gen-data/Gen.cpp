#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <limits>
#include <time.h> 
#include <math.h>
#include <sys/stat.h>
using namespace std;
using namespace cv;

ifstream& GotoLine(ifstream& file, unsigned int num);
void genImage(int mid, double slope);
void prepareData();

const int MAX_NUM = 1075902;
const char* POS_DISTRIBUTE_FILE = "gaussian/gaussian_121_60[1-272~265].txt";
const int MAX_SLOPE = 1186988; // MUST be bigger than MAX_NUM
const char* SLOPE_DISTRIBUTE_FILE = "gaussian/gaussian_30_60[-85-85].txt";

// texture must be square
const char* TEXTURE_FILE = "texture/wooden1.jpg";
const int TEXTURE_FILE_WIDTH = 2048;
const int TEXTURE_FILE_SIZE = 16*6; // cm

Mat texture;
Mat pattern;
int patternX, patternY;

int g_count = 1;

int main(int argc, char** argv )
{
    cout << "Preparing data\n";
    prepareData();

    if (argc != 2) {
        cout << "Arg is missing.\n";
        return -1;
    }

    char * pEnd;
    int n = strtol(argv[1], &pEnd, 10);

    if (n > MAX_NUM) {
        cout << "Arg must less than " << MAX_NUM << "\n";
        return -1;
    }

    cout << "Start gen image\n";
    ifstream datafile(POS_DISTRIBUTE_FILE);
    ifstream slopefile(SLOPE_DISTRIBUTE_FILE);
    srand(time(NULL));
    int dtpos = rand() % (MAX_NUM - n + 1) + 1;
    int slpos = rand() % (MAX_SLOPE - n + 1) + 1;
    GotoLine(datafile, dtpos);
    GotoLine(slopefile, slpos);

    int val;
    double sl;
    for (; g_count <= n; ++g_count) {
        slopefile >> sl;
        datafile >> val;

        genImage(val, sl);

        if (((g_count - 1) * 50) / n != (g_count * 50) / n) {
            cout << "\r[";
            for (int j = 0; j < 50; ++j) {
                cout << (j < (g_count * 50) / n ? (j == (g_count * 50) / n - 1 ? ">" : "=") : "-");
            }
            cout << "]" << flush;
        }
    }

    cout << "\nDONE\n";

    return 0;
}

void prepareData() {
    system("rm -r gen_image");
    mkdir("gen_image", S_IRWXU | S_IRWXG | S_IRWXO);
    for (int i = 0; i <= 272; ++i) {
        ostringstream ssNumber;
        ssNumber << "gen_image/" << i;
        mkdir(ssNumber.str().c_str(), S_IRWXU | S_IRWXG | S_IRWXO);
    }

    texture = imread(TEXTURE_FILE, CV_LOAD_IMAGE_UNCHANGED);
    double multiplier = 200 / TEXTURE_FILE_SIZE;
    double tP = TEXTURE_FILE_WIDTH * multiplier;

    double resizeArg = TEXTURE_FILE_WIDTH / TEXTURE_FILE_SIZE / 10;

    Mat multiTexture;
    repeat(texture, multiplier, multiplier, multiTexture);
    resize(multiTexture, multiTexture, Size(), 1 / resizeArg, 1 / resizeArg, INTER_NEAREST);
    tP /= resizeArg;

    Mat transformedTexture;
    Mat lambda(2, 4, CV_32FC1);
    lambda = Mat::zeros(multiTexture.rows, multiTexture.cols, multiTexture.type());

    // transform matrix
    Point2f pts1[4];
    Point2f pts2[4];

    pts1[0] = Point2f(0, 0);
    pts1[1] = Point2f(tP, 0);
    pts1[3] = Point2f(0, tP);
    pts1[2] = Point2f(tP, tP);

    pts2[0] = Point2f(tP / 4, tP * 3 / 4);
    pts2[1] = Point2f(tP * 3 / 4, tP * 3 / 4);
    pts2[3] = Point2f(0, tP);
    pts2[2] = Point2f(tP, tP);

    lambda = getPerspectiveTransform(pts1, pts2);
    warpPerspective(multiTexture, transformedTexture, lambda, transformedTexture.size());

    // crop
    Rect myROI(tP / 4, tP * 3 / 4, tP / 2, tP / 4);
    pattern = transformedTexture(myROI);
    patternX = pattern.cols;
    patternY = pattern.rows;
}

void genImage(int mid, double slope) {
    Mat out;
    int posx = rand() % (patternX / 2 - 272);
    int posy = rand() % (patternY - 18);

    Rect myROI(posx, posy, 272, 18);
    out = pattern(myROI).clone();

    Point2f top(mid + tan(slope * M_PI / 180) * 19, -10);
    Point2f bottom(mid - tan(slope * M_PI / 180) * 18, 28);
    int lthickness = rand() % 10 + 15;
    line(out, top, bottom, Scalar(255, 255, 255), lthickness);

    GaussianBlur(out, out, Size(5,5), 10);
    cvtColor(out, out, CV_BGR2GRAY);

    Mat noise = Mat(out.size(),CV_64F);
    Mat result;
    normalize(out, result, 0.0, 1.0, CV_MINMAX, CV_64F);
    imshow("OUTPUT0",result);
    randn(noise, 0, 0.1);
    imshow("noise",noise);
    imshow("out",out);
    result = result + noise;
    imshow("OUTPUT1",result);
    normalize(result, result, 0.0, 1.0, CV_MINMAX, CV_64F);
    imshow("OUTPUT",result);
    waitKey(0);

    ostringstream ssDirName;
    ssDirName << ((mid > 0 && mid <= 272) ? mid : 0);

    string dirName = ssDirName.str();

    ostringstream ss;
    ss << "gen_image/" << dirName << "/" << g_count << ".jpg";
    imwrite(ss.str(), out);
}

ifstream& GotoLine(ifstream& file, unsigned int num){
    file.seekg(ios::beg);
    for(int i=0; i < num - 1; ++i){
        file.ignore(numeric_limits<streamsize>::max(),'\n');
    }
    return file;
}