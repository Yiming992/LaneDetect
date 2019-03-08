#include "infer.hpp"

using namespace inference;
using namespace std; 
using namespace cv;
using namespace nvinfer1;

static Logger gLogger;

Mat Input_Reader::read(string f){
    Mat img_array=imread(f,IMREAD_UNCHANGED);
    Mat img_rgb;
    cvtColor(img_array,img_rgb,COLOR_BGR2RGB);
    return img_rgb;    
}

Mat Input_Reader::process(int w,int h,Mat t){
    if (Input_Reader::mode=="resize"){
        Mat processed_img;
        resize(t,processed_img,Size(w,h));
        return processed_img;
    }

const vector<string> directories{"../onnx_model/test.onnx"};

string NV_rt::locatefile(string input){
    return locateFile(input, directories);
}

void NV_rt::onnx2rt(string ModelFile, unsigned int BatchSize, IHostMemory* ModelStream){
    int verbosity=(int) ILogger::Severity::KWARNING;

    iBuilder* builder=createInferBuilder(gLogger);
    INetworkDefinition* network=builder->createNetwork();

    auto parser=nvonnxparser::createParser(*network,glogger);

    if (!parser->parseFromFile(locateFile(ModelFile,directories).c_str(),verbosity)){
        string msg("Failed to parse Onnx file")
        gLogger.log(nvinfer1::ILogger::Severity::kERROR,msg.c_str());
        exit(EXIT_FAILURE)
    }
    builder->setMaxBatchSize(BatchSize);
    builder->setMaxWorkspaceSize(1<<20);

    samplesCommon::enableDLA(builder,gUseDLACore);
    ICudaEngine* engine=builder->buildCudaEngine(*network);
    assert(engine);

    parser->destroy();

    ModelStream=engine->serialize();
    engine->destroy();
    network->destroy();
    builder->destroy();
}

void NV_rt::doInference(IExecutionContext& context,float* input,float* output,int batchsize){

    const ICudaEngine& engine=context.getEngine();

    assert(engine.getNbBindings()==2);
    void* buffers[2];
}




