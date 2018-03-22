
#include <cv.h>
#include <highgui.h>
#include <opencv2/opencv.hpp>

#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
using namespace cv;
using namespace std;

// For readTextFileIntoMatrix():
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <stdlib.h> // includes atof (for string to double conversion)

// Check if file exists:
#include <sys/stat.h>
#include <unistd.h>

// Gridding:
#include <math.h> // ceil()


// static void help()
// {
//     cout << "\nThis program demonstrates kmeans clustering from a text file.\n"
//             "Adapted from: https://docs.opencv.org/3.1.0/de/d63/kmeans_8cpp-example.html"
// }


// Declarations:
Mat readTextFileIntoMatrix(string path, char delimiter);
inline bool checkIfFileExists(const std::string& filename);
std::pair <float, float> minMaxOfMatCol(Mat points, int col_index);
//float minOfMatCol(Mat points, int colIndex);
//float maxOfMatCol(Mat points, int colIndex);
Mat makeEmptyGridFromMinxMinyMaxxMaxy(float x_min, float y_min, float x_max, float y_max, float resolution);


int main( int /*argc*/, char** /*argv*/ )
{
    
    // Get path to file from input:
    cout << "Enter path to text file to read in: \n";
    string path;
    cin >> path;
    cout << "Will read file from path " << path << "\n";

    cout << "Enter text file's delimiter character (, for example): \n";
    char delimiter;
    cin >> delimiter;
    cout << "File will be seperated by: " << delimiter << "\n";

    // Read input from delimited text file:
    Mat points = readTextFileIntoMatrix(path, delimiter);
    cout << "Input matrix: \n" << points;

    const int MAX_CLUSTERS = 5;
    Scalar colorTab[] =
    {
        Scalar(0, 0, 255),
        Scalar(0,255,0),
        Scalar(255,100,100),
        Scalar(255,0,255),
        Scalar(0,255,255)
    };
    // TODO: I am here: Need to generate an image (opencv class Mat) of the right dimensions
    // Get min and maxes:
    std::pair <float, float> x_min_max = minMaxOfMatCol(points, 0);
    std::pair <float, float> y_min_max = minMaxOfMatCol(points, 1);
    //float x_min = minOfMatCol(points, 0);
    //float y_min = minOfMatCol(points, 1);
    //float x_max = maxOfMatCol(points, 0);
    //float y_max = maxOfMatCol(points, 1);

    float resolution = 0.01;

    Mat img = makeEmptyGridFromMinxMinyMaxxMaxy(x_min_max.first, y_min_max.second, x_min_max.second, y_min_max.second, resolution);
    //Mat img(500, 500, CV_8UC3);
    RNG rng(12345);
    for(;;)
    {
        int k = 2; // Number of apriori estimated clusters.
        //clusterCount = 2;//, clusterCount = 2; //rng.uniform(2, MAX_CLUSTERS+1);
        //cout << "\nk = " << k << "\n";
        cout << "Number of clusters: = " << k << "\n";

        Mat labels;

        //int i, sampleCount = rng.uniform(1, 1001);
        // Make two opencv Mat objects, points and labels:
        //Mat points(sampleCount, 1, CV_32FC2), labels;
        //cout << "\n points from opencv: " << points;
        cout << "\n";



        //clusterCount = MIN(clusterCount, sampleCount);
        Mat centers;
        /* generate random sample from multigaussian distribution */
        //for( k = 0; k < clusterCount; k++ )
        //{
            // Initialize points:
            //Point center;
            //center.x = rng.uniform(0, img.cols);
            //center.y = rng.uniform(0, img.rows);
            /*Mat pointChunk = points.rowRange(k*sampleCount/clusterCount,
                                             k == clusterCount - 1 ? sampleCount :
                                             (k+1)*sampleCount/clusterCount); */
            //rng.fill(pointChunk, RNG::NORMAL, Scalar(center.x, center.y), Scalar(img.cols*0.05, img.rows*0.05));
        //}
        //randShuffle(points, 1, &rng);

        // Run kmeans:
        int numAttempts = 3; // Flag to specify the number of times the algorithm is executed using different initial labellings. The algorithm returns the labels that yield the best compactness (see the last function parameter).
        kmeans(points, k, labels,
            TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 10, 1.0),
               numAttempts, KMEANS_PP_CENTERS, centers);
        // KMEANS_PP_CENTERS: Use kmeans++ center initialization by Arthur and Vassilvitskii [Arthur2007].

        cout << "\n labels: \n " << labels ;
        cout << "\n" ;

        // Plot kmeans:
        int sampleCount = points.rows;
        img = Scalar::all(0);
        int i;
        for( i = 0; i < sampleCount; i++ )
        {
            int clusterIdx = labels.at<int>(i);
            Point ipt = points.at<Point2f>(i);
            circle( img, ipt, 2, colorTab[clusterIdx], FILLED, LINE_AA );
        }
        imshow("clusters", img);
        char key = (char)waitKey();
        if( key == 27 || key == 'q' || key == 'Q' ) // 'ESC'
            break;
    }
    return 0;
}

Mat readTextFileIntoMatrix(string path, char delimiter) {
    /*
    Reads delimited text file into OpenCV Mat.
    All entries in delimited file must be numeric. Function assumes first line is header (which is currenlty just skipped).
     */

    // check to make sure file is open:
    if (checkIfFileExists(path)) {
        cout << "File " << path << " exists.";
    } else {
        cout << "File" << path << " does not exist.";
    }

    ifstream file(path);

    if (!file.is_open())
    {
        std::exit(EXIT_FAILURE);
    }

    Mat matrix;

    // Skip the first line
    std::string firstLine;
    std::getline(file, firstLine);

    int numRows = 0;
    int numCells = 0;
    std::string line;
    while (std::getline(file, line)) {  
        //cout << "\n std::getline \n";
        istringstream stream(line);
        cout << line;
        //char sep; //comma!
        std::string cellString;
        //char* end;
        // read *both* a number and the delimiter:
        while(std::getline(stream, cellString, delimiter)) {
            //cout << "\n stream \n";
            //cout << "cellString: " << cellString;

            float cell = atof(cellString.c_str());

            matrix.push_back(cell);

            numCells ++;
        }
        numRows ++;
    } 
    cout << "\n numRows: " << numRows;
    cout << "\n number of cells/elements: " << numCells; 
    // reshape to 2d:
    matrix = matrix.reshape(1,numRows);

    //cout << "\n Matrix in function: \n" << matrix;

    return matrix;
}

inline bool checkIfFileExists(const std::string& filename) {
    //https://stackoverflow.com/questions/12774207/fastest-way-to-check-if-a-file-exist-using-standard-c-c11-c
    struct stat buffer;   
    return (stat (filename.c_str(), &buffer) == 0); 
}


std::pair <float, float> minMaxOfMatCol(Mat points, int col_index) {
    // Returns the min and max values of a column of an opencv Mat 
    std::pair <float, float> min_max;

    double min_val;
    double max_val;
    Point min_loc;
    Point max_loc;
    minMaxLoc(points.col(col_index), &min_val, &max_val, &min_loc, &max_loc);

    min_max = std::make_pair((float) (min_val), (float) (max_val));
    return min_max;
}

Mat makeEmptyGridFromMinxMinyMaxxMaxy(float x_min, float y_min, float x_max, float y_max, float resolution) {
    
    // add one for column and row at right edge and bottom, respectively
    n_rows = static_cast<int>((ceilToRes(y_max, resolution) - floorToRes(y_min, resolution))/resolution) + 1;

    n_cols = static_cast<int>((ceilToRes(x_max, resolution) - floorToRes(x_min, resolution))/resolution) + 1;

    return Mat img(n_rows, n_cols, CV_8UC3);
}

float ceilToRes(float x, float resolution) {
    return ceil(x / resolution) * resolution;
}

float floorToRes(float x, float resolution) {
    // floors x to resolution
    return floor(x / resolution) * resolution;
}

int rowIndex(float y, float y_max, float resolution) {
    // Computes the row index in a grid given a real value, y
    return static_cast<int>((y_max - y) / resolution);
}

int columnIndex(float x, float x_min, float resolution) {
    // Computes the column index in a grid given a real value, x
    return static_cast<int>((x - x_min) / resolution);
}




// here's another example: https://stackoverflow.com/questions/33250380/c-skip-first-line-of-csv-file
/*
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <cstdlib>

int readCSV() {     
    std::ifstream data("prisoner_data.csv");
    if (!data.is_open())
    {
        std::exit(EXIT_FAILURE);
    }
    std::string str;
    std::getline(data, str); // skip the first line
    while (std::getline(data, str))
    {
        std::istringstream iss(str);
        std::string token;
        while (std::getline(iss, token, ','))
        {   
            // process each token
            std::cout << token << " ";
        }
        std::cout << std::endl;
    }
}*/
