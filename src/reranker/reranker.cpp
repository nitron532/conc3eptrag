#include "reranker.h"

#if !defined(SAFETENSORS_CPP_NO_IMPLEMENTATION)
#define SAFETENSORS_CPP_IMPLEMENTATION
#endif
#include "safetensors.hh"

#define USE_MMAP

double cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b){
    float dot = 0, na = 0, nb = 0;
        for (int i = 0; i < (int)a.size(); i++) {
            dot += a[i] * b[i];
            na  += a[i] * a[i];
            nb  += b[i] * b[i];
        }
    return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-8f);
}

ReRanker::ReRanker(size_t gpuLayers, const char* rerankingModelPath){

        llama_backend_init();
        this->modelParams = llama_model_default_params();
        this->modelParams.n_gpu_layers = gpuLayers;
        this->rerankingModelPath = rerankingModelPath;

        this->rerankingModel = llama_model_load_from_file(
            rerankingModelPath,
            modelParams
        );

        if(!rerankingModel){
            std::cerr << ("Couldn't find model at specified path.\n") << std::endl;
            exit(1);
        }
        
        this->contextParams = llama_context_default_params(); //consider splitting into multiple contexts and multithreading inference
        this->contextParams.embeddings = true;
        this->contextParams.pooling_type = LLAMA_POOLING_TYPE_RANK;
        this->context = llama_init_from_model(rerankingModel, contextParams);
        this->vocab = llama_model_get_vocab(rerankingModel);

    }

ReRanker::~ReRanker(){
        llama_free(this->context);
        llama_model_free(this->rerankingModel);
        llama_backend_free();
    }

void ReRanker::modelReRank(){
    if(this->query->size() == 0 || this->docs->size() == 0){
        std::cerr << "Error: Either provided documents or queries are of length 0." << std::endl;
        return;
    }

    std::string systemPrompt = "<|im_start|>system\nYou are a search relevance expert who can determine a ranking of documents based on their relevance to the query.\n<|im_end|>";
    std::string userPrompt = "<|im_start|>user\nI will provide you with " + std::to_string(this->docs->size()) + 
    " passages, each indicated by a numerical identifier. Rank the passages based on their relevance to query:" + *(this->query) + '\n';

    for(size_t i = 0; i < this->docs->size(); i++){
        userPrompt += "<passage id=\"" + std::to_string(i) + "\">\n" + (*(this->docs))[i] + "<|doc_emb|>\n</passage>\n";
    }
    userPrompt += "\n<query>\n" + *(this->query) + "<|query_emb|>\n</query>\n<|im_end|>";

    llama_token docEmbbeddingId, queryEmbeddingId;
    {
        std::vector<llama_token> tmp(8);
        int n = llama_tokenize(vocab, "<|doc_emb|>", 11, tmp.data(), 8, false, true);
        docEmbbeddingId = tmp[0];
        n = llama_tokenize(vocab, "<|query_emb|>", 13, tmp.data(), 8, false, true);
        queryEmbeddingId = tmp[0];
    }
    std::string fullPrompt = systemPrompt + userPrompt;
    std::vector<llama_token> tokens(fullPrompt.size() + 64); //buffer size leg room

    size_t numTokens = llama_tokenize(this->vocab, fullPrompt.c_str(), fullPrompt.size(), tokens.data(), tokens.size(), true, true);
    tokens.resize(numTokens);
    std::vector<size_t> docEmbeddingPositions;
    size_t queryEmbeddingPosition = -1;
    for (size_t i = 0; i < numTokens; i++) {
        if (tokens[i] == docEmbbeddingId) {docEmbeddingPositions.push_back(i);}
        if (tokens[i] == queryEmbeddingId) {queryEmbeddingPosition = i;}
    }

    llama_batch batch = llama_batch_init(numTokens,0,1);

    for (int i = 0; i < numTokens; i++) {
        batch.token[i]     = tokens[i];
        batch.pos[i]       = i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;

        batch.logits[i] = (tokens[i] == docEmbbeddingId || tokens[i] == queryEmbeddingId) ? 1 : 0;
    }
    batch.n_tokens = numTokens;
    llama_decode(context, batch);
    llama_batch_free(batch);
    llama_memory_seq_rm(llama_get_memory(this->context), sequenceId, -1,-1); //clear last sequence's kv cache values
    this->sequenceId++;

    size_t numEmbeddings = llama_model_n_embd(this->rerankingModel);

    float* queryEmbedding = llama_get_embeddings_ith(context, queryEmbeddingPosition);
    std::vector<float> queryEmbeddingVector(queryEmbedding, queryEmbedding + numEmbeddings);

    std::vector<std::vector<float>> docEmbeddingVectors;

    try{
        safetensors::safetensors_t st = loadSafeTensors("projector.safetensors"); //jina specific
        //project floats through safe tensors
        

    } catch (const std::exception& e){
        std::cerr << "Failed to load safe tensors from projector file: "<< e.what() << "\nUsing full sized non-projected embeddings." << std::endl;

        for(size_t pos : docEmbeddingPositions){
            float* emb = llama_get_embeddings_ith(this->context, pos);
            docEmbeddingVectors.push_back(std::vector<float>(emb, emb + numEmbeddings));
        }

    }

    for(size_t i = 0; i < docEmbeddingVectors.size(); i++){
        this->rankings.push_back({cosineSimilarity(docEmbeddingVectors[i],queryEmbeddingVector),i});
    }
    std::sort(this->rankings.begin(), this->rankings.end());
    
}

void ReRanker::setQueryAndDocuments(std::string& query, std::vector<std::string>& documents){
    this->query = &query;
    this->docs = &documents;
}

const std::vector<std::pair<float,int>> * ReRanker::getRankings(){
    return &rankings;
}


safetensors::safetensors_t loadSafeTensors(const char* projectorPath){
    safetensors::safetensors_t st;
    std::string projectorPath = "projector.safetensors";
    std::string warn, err;

    #if defined(USE_MMAP)
        printf("USE mmap\n");
        bool ret = safetensors::mmap_from_file(projectorPath, &st, &warn, &err);
    #else
        bool ret = safetensors::load_from_file(projectorPath, &st, &warn, &err);
    #endif

    if (warn.size()) {
        std::cout << "WARN: " << warn << "\n";
    }

    if (!ret) {
        throw std::runtime_error("Failed to load with projector weights with error: " + err);
    }

    if (!safetensors::validate_data_offsets(st, err)) {
        throw std::runtime_error("Invalid data offsets: " + err);
    }

    return st;
}