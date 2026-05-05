#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <cmath>

#include <llama-cpp.h>
#include <llama.h>


class ReRanker{
    private:
        const std::string* query;
        const std::vector<std::string>* docs;
        std::vector<std::pair<float,int>> rankings; //index 0 -> index of document of best similarity with score
        
        llama_model_params modelParams;
        size_t gpuLayers = 0;
        const char* rerankingModelPath; //"jina-reranker-v3-Q8_0.gguf"
        llama_model* rerankingModel;
        llama_context_params contextParams;
        llama_context* context;
        const llama_vocab* vocab;
        size_t sequenceId = 0;

    public:
        ReRanker(size_t gpuLayers, const char* rerankingModelPath);

        ~ReRanker();

        void modelReRank();

        void setQueryAndDocuments(std::string& query, std::vector<std::string>& documents);

        const std::vector<std::pair<float,int>> * getRankings();

        const std::string * getQuery();
        const std::vector<std::string> * getDocuments();

};