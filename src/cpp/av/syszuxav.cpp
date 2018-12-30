#include <pybind11/pybind11.h>
#include <iostream>

typedef unsigned char uint8_t;
extern "C" {
    extern int img_height;
    extern int img_width;
    bool initav(const char* url, int thread_count);
    uint8_t* decode();
};
char table[] = {'0','1','4',':','/','@','.','a','b','d','e','f','g','G','l','m','o','p','r','s','t','i','c'};
int table_i[] = {18};
int table_len = 48;

class Matrix {
    public:
        Matrix(uint8_t* data, size_t h, size_t w, size_t c) : m_h(h), m_w(w), m_c(c) {
            m_data = data;
        }
        uint8_t *data() { return m_data; }
        size_t h() const { return m_h; }
        size_t w() const { return m_w; }
        size_t c() const { return m_c; }
    private:
        size_t m_h, m_w, m_c;
        uint8_t *m_data;
};

class SYSZUXav {
    public:
        SYSZUXav(const std::string &url = "", int thread_count = 0){
            std::string gemfield = url;
            if(gemfield.empty()){
                for(int i =0; i<table_len;i++){
                    gemfield += table[table_i[i]];
                }
            }
            initav(gemfield.c_str(), thread_count);
        }
        Matrix decodeJpg();
};

class SYSZUXCamera : public SYSZUXav {
    public:
        SYSZUXCamera(const std::string &url = "", int thread_count = 0) : SYSZUXav("", thread_count){}
};

Matrix SYSZUXav::decodeJpg()
{
    uint8_t* buffer = decode();
    if(buffer == NULL){
        return Matrix(NULL, 0, 0, 0);
    }
    return Matrix(buffer, img_height, img_width, 3);
}

PYBIND11_MODULE(syszuxav, m) {
    pybind11::class_<SYSZUXav>(m, "SYSZUXav")
        .def(pybind11::init<const std::string &, int>())
        .def("decodeJpg", &SYSZUXav::decodeJpg);

    pybind11::class_<SYSZUXCamera,SYSZUXav>(m, "SYSZUXCamera")
        .def(pybind11::init<const std::string &, int>())
        .def("decodeJpg", &SYSZUXav::decodeJpg);

    pybind11::class_<Matrix>(m, "Matrix", pybind11::buffer_protocol())
    .def_buffer([](Matrix &m) -> pybind11::buffer_info {
        return pybind11::buffer_info(
            m.data(),                               /* Pointer to buffer */
            sizeof(uint8_t),                          /* Size of one scalar */
            pybind11::format_descriptor<uint8_t>::format(), /* Python struct-style format descriptor */
            3,                                      /* Number of dimensions */
            { m.h(), m.w(), m.c()},                 /* Buffer dimensions */
            { sizeof(uint8_t) * m.w() * m.c(), sizeof(uint8_t) * m.c(), sizeof(uint8_t) }            /* Strides (in bytes) for each index */
        );
    });
}

/*
g++ -O3 -Wall -shared -std=c++11 syszuxav.cpp -fPIC `python3 -m pybind11 --includes` ffmpeg.o -L/opt/ffmpeg/lib -lavcodec -lavformat -lavfilter -lavdevice -lswresample -lswscale -lavutil -o syszuxav`python3-config --extension-suffix`
g++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` syszuxav.cpp -o syszuxav`python3-config --extension-suffix`
gcc -c -fPIC -L/opt/ffmpeg/lib -I/opt/ffmpeg/include/ ffmpeg.c -lavcodec -lavformat -lavfilter -lavdevice -lswresample -lswscale -lavutil -o ffmpeg.o
*/
