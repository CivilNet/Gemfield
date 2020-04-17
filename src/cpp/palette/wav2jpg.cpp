/*
 * Convert wav file to jpg based on Essentia.
 */

#include <iostream>
#include <fstream>
#include <essentia/algorithmfactory.h>
#include <essentia/pool.h>
#include "syszux_palette.h"
#include <experimental/filesystem>
using namespace std;
using namespace essentia;
using namespace standard;

int main(int argc, char* argv[]) {

  if (argc != 4) {
    cout << "Error: incorrect number of arguments." << endl;
    cout << "Usage: " << argv[0] << " audio_input_directory num2cut channel_mode" << endl;
    exit(1);
  }

  string audioFilenameDir = argv[1];
  string num2cut_str = argv[2];
  string channel_mode = argv[3];
  int num2cut = std::stoi(num2cut_str);

  essentia::init();
  for(const auto& dirEntry:std::experimental::filesystem::recursive_directory_iterator(audioFilenameDir)){
    std::cout<<"gemfield: "<<dirEntry.path()<<std::endl;
    std::string audioFilename = dirEntry.path();

    if(std::experimental::filesystem::path(audioFilename).extension() != ".wav"){
      continue;
    }
    string outputFilename = audioFilenameDir + "/gemfield/" +audioFilename.substr(audioFilenameDir.length()) + "_opencv_";
    string outputDir = std::experimental::filesystem::path(outputFilename).parent_path();
    if (!std::experimental::filesystem::exists(outputDir)) {
      std::experimental::filesystem::create_directories(outputDir);
      std::cout<<"Gemfield create directory: "<<outputDir<<std::endl;
    }
///////////////////////////////////////////////////GEMFIELD TEST///////

    float output_sr = 16000.0;
    int fs = 400.0;
    int hs = 200.0;
    AlgorithmFactory& factory = AlgorithmFactory::instance();
    //Algorithm* resampler = AlgorithmFactory::create("Resample");
    std::unique_ptr<Algorithm>  audioLoader( factory.create("MonoLoader","filename", audioFilename,"sampleRate", 44100,"downmix", channel_mode));
    std::unique_ptr<Algorithm> resampler_up(AlgorithmFactory::create("Resample"));
    resampler_up->configure("inputSampleRate", 44100,"outputSampleRate", output_sr,"quality", 0);
    
    //Algorithm* frameCutter = factory.create("FrameCutter","frameSize", fs,"hopSize", hs,"startFromZero", true);
    std::unique_ptr<Algorithm> frameCutter_up(factory.create("FrameCutter","frameSize", fs,"hopSize", hs,"startFromZero", true));
    
    //Algorithm* vggish = factory.create("TensorflowInputVGGish");
    std::unique_ptr<Algorithm> vggish_up(factory.create("TensorflowInputVGGish"));
    //from iOS AVAssetReader
    vector<float> audio;
    vector<float> audio_res;
    vector<float> frame;
    vector<float> vggish_output;

    audioLoader->output("audio").set(audio);
    resampler_up->input("signal").set(audio);
    resampler_up->output("signal").set(audio_res);

    frameCutter_up->input("signal").set(audio_res);
    frameCutter_up->output("frame").set(frame);

    vggish_up->input("frame").set(frame);
    vggish_up->output("bands").set(vggish_output);

    audioLoader->compute();
    resampler_up->compute();

    int gemfield = 0;
    vector<vector<float> > rc;
    vector<vector<float> > vec2cut;
    vector<unique_ptr<unsigned char[]> > rgb_array;
    vector<unique_ptr<unsigned char[]> > rgb_array_round;
    std::cout<<"gemfield debug num2cut: "<<num2cut<<std::endl;
    while (true) {
        // compute a frame
        frameCutter_up->compute();
        if (!frame.size()) {
            break;
        }
        vggish_up->compute();
        vec2cut.push_back(vggish_output);
        gemfield ++;
        if(vec2cut.size() == num2cut){
            //rgb_array.push_back(syszuxpalette::createRGBArrayFromMatrix(vec2cut));
            rgb_array.push_back(syszuxpalette::createRGBArrayFromMatrix(vec2cut, syszuxpalette::ImageMode::BGR));
	    rgb_array_round.push_back(syszuxpalette::createRGBArrayFromMatrix(vec2cut, syszuxpalette::ImageMode::BGR,true,true));
            //50% overlap
            //vec2cut.erase(vec2cut.begin() , vec2cut.begin()+ (num2cut+1)/2);
	    //0.25s overlap , 1s == 80
	    vec2cut.erase(vec2cut.begin() , vec2cut.begin()+20);
        }
	if(rgb_array.size() == 5){
	  break;
	}
    }
    std::cout<<"gemfield debug rgb array size: "<<rgb_array.size()<<std::endl;
    for(auto &x : rgb_array){
      break;
      for(int i=0;i<num2cut*64*3;i++){
        if(i%(num2cut*3) == 0){
          std::cout<<std::endl;
          std::cout<<"----------------GEMFIELD"<<int(i/(num2cut*3))<<"----------------------"<<std::endl;
        }else if(i%3==0){
          std::cout<<std::endl;
        }
        std::cout<<(unsigned int)x[i]<<",";
      }
      break;
    }
    std::cout<<"generate file: "<<outputFilename<<std::endl;
    syszuxpalette::syszuxMultiImgWrite(outputFilename, rgb_array,64 ,num2cut);
    syszuxpalette::syszuxMultiImgWrite(outputFilename + "round_", rgb_array_round,64 ,num2cut);
//////////////////////////////////////////////////GEMFIELD TEST////////
  }
  essentia::shutdown();
  return 0;
}


