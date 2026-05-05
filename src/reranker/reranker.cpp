#include "reranker.h"

double cosineSimilarity(const float* a, const float* b, size_t size){
    float dot = 0, na = 0, nb = 0;
        for (int i = 0; i < size; i++) {
            dot += a[i] * b[i];
            na  += a[i] * a[i];
            nb  += b[i] * b[i];
        }
    return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-8f);
}

void ReRanker::loadSafeTensors(const char* projectorPath){
    // std::string projectorPath = "projector.safetensors";
    std::string warn, err;

    #if defined(USE_MMAP)
        printf("USE mmap\n");
        bool ret = safetensors::mmap_from_file(projectorPath, &this->st, &warn, &err);
    #else
        bool ret = safetensors::load_from_file(projectorPath, &st, &warn, &err);
    #endif

    if (warn.size()) {
        std::cout << "WARN: " << warn << "\n";
    }

    if (!ret) {
        throw std::runtime_error("Failed to load with projector weights with error: " + err);
    }

    if (!safetensors::validate_data_offsets(this->st, err)) {
        throw std::runtime_error("Invalid data offsets: " + err);
    }

    const uint8_t* databuffer = nullptr;
    this->databuffer = this->st.mmaped ? this->st.databuffer_addr : this->st.storage.data();
}

const std::pair<const float*, std::vector<size_t>> ReRanker::getTensor(const char* name) {
        safetensors::tensor_t t;
        st.tensors.at(name, &t);
        const float* data = reinterpret_cast<const float*>(this->databuffer + t.data_offsets[0]);
        return {data, t.shape};
}

const float* ReRanker::linearLayer(const float* in, const float* weight, const float* bias, size_t inDim, size_t outDim, bool relu){
    float* output = new float[outDim];
    for(size_t i = 0; i < outDim; i++){
        float sum = bias ? bias[i] : 0.0f;
        for(size_t j = 0; j < inDim; j++){
            sum += weight[i * inDim + j] * in[j];
        }
        output[i] = relu ? std::max(0.0f, sum) : sum; //relu
    }
    return output;
}

const float* ReRanker::project(const float* embedding){ //1024 dimensional raw embedding
    const std::pair<const float*, std::vector<size_t>> firstLayer = getTensor("projector.0.weight");
    const std::pair<const float*, std::vector<size_t>> secondLayer = getTensor("projector.2.weight");
    
    const float* firstActivation = linearLayer(embedding, firstLayer.first, nullptr, firstLayer.second[1], firstLayer.second[0], true);
    const float* secondActivation = linearLayer(firstActivation, secondLayer.first, nullptr, secondLayer.second[0], secondLayer.second[1], false);
    delete [] firstActivation;
    return secondActivation;
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
        this->contextParams.pooling_type = LLAMA_POOLING_TYPE_NONE; //model uses lastbutnotlate, so there are specific tokens that need extraction
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
        userPrompt += "<passage id=\"" + std::to_string(i) + "\">\n" + (*(this->docs))[i] + "<|rerank_token|>\n</passage>\n";
    }
    userPrompt += "\n<query>\n" + *(this->query) + "<|embed_token|>\n</query>\n<|im_end|>";

    llama_token docEmbeddingId, queryEmbeddingId;
    {
        //gguf model uses different embedding token ids and names than listed in jina reranker v3 paper.
        std::vector<llama_token> tmp(1);
        int n = llama_tokenize(this->vocab, "<|embed_token|>", 15, tmp.data(), 1, false, true); //should be 151670
        queryEmbeddingId = tmp[0];
        n = llama_tokenize(this->vocab, "<|rerank_token|>", 16, tmp.data(), 1, false, true); //should be 151671
        docEmbeddingId = tmp[0];
    }

    std::string fullPrompt = systemPrompt + userPrompt;
    std::vector<llama_token> tokens(fullPrompt.size() + 64); //buffer size leg room

    size_t numTokens = llama_tokenize(this->vocab, fullPrompt.c_str(), fullPrompt.size(), tokens.data(), tokens.size(), true, true);
    tokens.resize(numTokens);

    size_t* docEmbeddingPositions = new size_t[this->docs->size()];
    size_t queryEmbeddingPosition = 0;
    size_t j = 0;
    for (size_t i = 0; i < numTokens; i++) {
        if (tokens[i] == docEmbeddingId) {docEmbeddingPositions[j++] = i;}
        else if (tokens[i] == queryEmbeddingId) {queryEmbeddingPosition = i;}
    }
    
    llama_batch batch = llama_batch_init(numTokens,0,1);

    for (int i = 0; i < numTokens; i++) {
        batch.token[i]     = tokens[i];
        batch.pos[i]       = i;
        batch.n_seq_id[i]  = 1; //how many sequences this token belongs to
        batch.seq_id[i][0] = this->sequenceId; //which sequence it belongs to

        batch.logits[i] = (tokens[i] == docEmbeddingId || tokens[i] == queryEmbeddingId) ? 1 : 0; //mark 1 for output
    }
    batch.n_tokens = numTokens;
    llama_decode(context, batch);
    llama_batch_free(batch);
    llama_memory_seq_rm(llama_get_memory(this->context), sequenceId, -1,-1); //clear last sequence's kv cache values.
    this->sequenceId++;

    size_t rawEmbeddingDimension = llama_model_n_embd(this->rerankingModel); //1024 dimensional embeddings, these will be projected using projector.safetensor
    const float* queryEmbeddingVector;

    std::vector<const float*> docEmbeddingVectors;
    size_t finalEmbeddingDimension;

    try{
        loadSafeTensors("projector.safetensors");
        for(size_t i = 0; i < this->docs->size(); i++){
            docEmbeddingVectors.push_back(project(llama_get_embeddings_ith(this->context, docEmbeddingPositions[i])));
        }
        queryEmbeddingVector = project(llama_get_embeddings_ith(this->context, queryEmbeddingPosition));
        finalEmbeddingDimension = 512;
        //jina reranker v3 paper says 256, but gguf model version's projection.safetensors only has two layers that project to 512

    } catch (const std::exception& e){
        std::cerr << "Failed to load safe tensors from projector file: "<< e.what() << "\nUsing full sized non-projected embeddings." << std::endl;
        for(size_t i = 0; i < this->docs->size(); i++){
            docEmbeddingVectors.push_back(llama_get_embeddings_ith(this->context, docEmbeddingPositions[i]));
        } 
        queryEmbeddingVector = llama_get_embeddings_ith(this->context, queryEmbeddingPosition);
        finalEmbeddingDimension = 1024;
    }

    for(size_t i = 0; i < docEmbeddingVectors.size(); i++){
        this->rankings.push_back({cosineSimilarity(docEmbeddingVectors[i],queryEmbeddingVector, finalEmbeddingDimension),i});
    }
    std::sort(this->rankings.begin(), this->rankings.end(), std::greater<>());

    delete [] docEmbeddingPositions;
    for (const float* e : docEmbeddingVectors){
        delete [] e;
    }
}

void ReRanker::setQueryAndDocuments(std::string& query, std::vector<std::string>& documents){
    this->query = &query;
    this->docs = &documents;
}
const std::string * ReRanker::getQuery(){
    return this->query;
}

const std::vector<std::string> * ReRanker::getDocuments(){
    return this->docs;
}

const std::vector<std::pair<float,int>> * ReRanker::getRankings(){
    return &this->rankings;
}