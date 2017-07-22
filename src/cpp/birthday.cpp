#include <iostream>
#include <string.h>

void usage(const std::string& program, const std::string& info = ""){
    std::cout<<"Usage: "<<program<<" <person_number>"<<std::endl;
    if(!info.empty()){
        std::cout<<info<<std::endl;
    }
}
long double predict(int persons){
    const long double DAYS = 365.0;
    long double prediction = 1.0;
    if(persons >= DAYS){
        return 1;
    }
    //first calculate the probability that none persons have the same personday
    for(int i = 0; i < persons; i++){
        prediction *= (DAYS - i)/DAYS;
    }
    //Second we got the probability that at least 2 persons have the same birthday
    return 1 - prediction;
}
int main(int argc, char** argv)
{
    if(argc < 2){
        usage(argv[0]);
        return 1;
    }
    int persons = 0;
    try{
        persons = std::stoi(argv[1]);
    }catch(std::exception& e){
        usage(argv[0], strcat("Error: ", e.what()));
        return 2;
    }catch(...){
        usage(argv[0], "Unknown error");
        return 3;
    }
    std::cout<<"The probability that at least 2 persons in "<<persons<<" have the same birthday is: "<< predict(persons)<< std::endl;
    return 0;
}
