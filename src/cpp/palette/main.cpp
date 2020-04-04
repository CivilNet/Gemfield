#include <iostream>
#include "syszux_palette.h"
using namespace std;
//g++ -DSYSZUX_PALETTE_WITH_OPENCV main.cpp -o main -lopencv_core -lopencv_imgcodecs
int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " output_path" << std::endl;
        return 1;
    }

    const std::string output_path = argv[1];

    constexpr int width  = 760;
    constexpr int height = 60;

    vector<vector<float> > matrix = {};

    for (int x = 0; x < width; ++ x)
    {
        vector<float> h_vec; 
        const double value = static_cast<double>(x) / static_cast<double>(width - 1);
        for (int y = 0; y < height; ++ y)
        {
            h_vec.push_back(value);
        }
        matrix.push_back(h_vec);
    }
    unique_ptr<unsigned char[]> rc = syszuxpalette::createRGBArrayFromMatrix(matrix, syszuxpalette::ImageMode::BGR);

#if defined(SYSZUX_PALETTE_WITH_OPENCV)
    //visualize
    syszuxpalette::syszuxImgWrite(output_path, rc, matrix[0].size(),matrix.size());
#endif
    return 0;
}