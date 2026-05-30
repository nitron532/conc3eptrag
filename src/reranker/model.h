#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <cmath>

#include <llama-cpp.h>
#include <llama.h>

#define SAFETENSORS_CPP_NO_IMPLEMENTATION
#include "safetensors.hh"
#define USE_MMAP

//TODO change to abstract base type model and reranker and embedder are child classes to remove redundant code
//designing non smartly to quickly prototype since code should mostly be the same

class Model{
    private:
        const std::string* query;
        const std::vector<std::string>* docs;
        std::vector<std::pair<float,int>> rankings; //index 0 -> index of document of best similarity with score
        std::vector<std::vector<float>> embedding; 


        llama_model_params modelParams;
        size_t gpuLayers = 0;
        const char* modelPath;
        llama_model* model;
        llama_context_params contextParams;
        llama_context* context;
        const llama_vocab* vocab;
        size_t sequenceId = 0;

        safetensors::safetensors_t st;
        const uint8_t* databuffer = nullptr;

        void loadSafeTensors(const char* projectorPath);
        const std::pair<const float*, std::vector<size_t>> getTensor(const char* name);
        const float* linearLayer(const float* in, const float* weight, const float* bias, size_t inDim, size_t outDim, bool relu);
        const float* project(const float* embedding);

    public:
        Model(size_t gpuLayers, const char* modelPath, bool embedder);

        ~Model();

        void modelEmbed(bool query);

        void modelReRank();

        void setQueryAndDocuments(std::string& query, std::vector<std::string>& documents);

        const std::vector<std::pair<float,int>> * getRankings();

        const std::vector<float> * getEmbedding();

        const std::string * getQuery();
        const std::vector<std::string> * getDocuments();

};