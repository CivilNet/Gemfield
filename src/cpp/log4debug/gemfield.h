#ifndef GEMFIELD
#define GEMFIELD 1
#include <initializer_list>
#include <iostream>
#include <sstream>
#include <mutex>
#include <thread>
#include <map>

#define GEMFIELDSTR_DETAIL(x) #x
#define GEMFIELDSTR(x) GEMFIELDSTR_DETAIL(x)

thread_local int __attribute__((weak)) gemfield_counter = 0;
//std::map<uint64_t, int> __attribute__((weak)) gemfield_counter_map;

class Gemfield{
public:
    Gemfield(std::initializer_list<const char*> src){
        std::stringstream ss;
        ss << std::this_thread::get_id();
        
        s += "[";
        s += ss.str();
        s += "]";
        for (auto s1: src){
            s += ":";
            s += s1;
        }
         
        printMark('+', s);
    }
    ~Gemfield(){
        printMark('-', s);
    }

private:
    static void printMark(char c, std::string& s){
        static std::mutex gemfield_lock;
        std::lock_guard<std::mutex> lock(gemfield_lock);

        //std::thread::id current_tid = std::this_thread::get_id();
        std::stringstream ss;
        ss << std::this_thread::get_id();
        uint64_t current_tid = std::stoull(ss.str());

        if(c == '+'){
            ++gemfield_counter;
        }
        for(int i=0; i< gemfield_counter; i++){
            std::cout<<c;
        }
        std::cout<<s<<std::endl;

        if(c == '-'){
            --gemfield_counter;
        }
    }
    std::string s;
};
#endif
