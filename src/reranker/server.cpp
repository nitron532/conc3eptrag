#include "reranker.h"
#include <asio.hpp>
#include <unordered_map>
#include <fstream>

using asio::ip::tcp;

// #define DEBUG

int main(int argc, char *argv[])
{
    if (argc < 2) {
        std::cerr << "Usage: ./rerankerserver <portnumber>" << std::endl;
        exit(1);
    }
    int portNum = atoi(argv[1]);

    //initialize llama.cpp reranker
    ReRanker jina(0,"jina-reranker-v3-Q8_0.gguf");
    
#ifdef DEBUG
    std::string q = "What are the health benefits of green tea?";
    std::vector<std::string> documents = {
    "Green tea contains antioxidants called catechins that may help reduce inflammation and protect cells from damage.",
    "El precio del café ha aumentado un 20% este año debido a problemas en la cadena de suministro.",
    "Studies show that drinking green tea regularly can improve brain function and boost metabolism.",
    "Basketball is one of the most popular sports in the United States.",
    "绿茶富含儿茶素等抗氧化剂，可以降低心脏病风险，还有助于控制体重。",
    "Le thé vert est riche en antioxydants et peut améliorer la fonction cérébrale.",
    };

    // // 0.461935 0
    // // 0.350195 4
    // // 0.254084 5
    // // 0.235676 2
    // // -0.128770 1
    // // -0.156261 3
    
    jina.setQueryAndDocuments(q, documents);
    jina.modelReRank();
    const std::vector<std::pair<float,int>> * rankings = jina.getRankings();

    for(int i = 0; i < rankings->size(); i++){
        std::cout << std::to_string((*rankings)[i].first) << " " << std::to_string((*rankings)[i].second) << std::endl;
    }

    return 0;
#endif

    asio::io_context ioContext;
    tcp::acceptor acceptor(ioContext, tcp::endpoint(tcp::v4(), portNum));
    std::error_code ignoredError;
    std::string_view delim = "$EOM$";

    while(true){ //listening loop
        asio::streambuf readbuffer;
        tcp::socket socket(ioContext);
        std::cout << "Listening on: "<< portNum << std::endl;
        acceptor.accept(socket);
        size_t clientPort = socket.remote_endpoint().port();
        std::cout << "Connected to " << clientPort << std::endl;
        std::string message;
        size_t readNum;
        std::string response = "Server at " + socket.local_endpoint().address().to_string() + ":" + std::to_string(socket.local_endpoint().port());
        while(true){ //single client loop
            try{
                readNum = asio::read_until(socket, readbuffer, delim);
                message = std::string(asio::buffers_begin(readbuffer.data()),asio::buffers_begin(readbuffer.data())+readNum);
                readbuffer.consume(readNum);
                std::cout << "received: " << message << std::endl;
                message = message.substr(0, message.size()-5);
                if(message.substr(message.rfind('.')) == ".txt"){
                    response += " will rerank " + message + "$EOM$";
                    asio::write(socket, asio::buffer(response), ignoredError);
                    std::vector<std::string> documents;
                    std::string query;
                    std::ifstream infile(message);
                    std::string line;
                    std::vector<size_t> posToId;
                    size_t i = 0;
                    while(std::getline(infile, line)){
                        if(!isdigit(line[0])){
                            query = line;
                        }
                        else{
                            size_t contextPos = line.find_first_not_of("0123456789");
                            documents.push_back(line.substr(contextPos+1));
                            posToId.push_back(stoi(line.substr(0, contextPos)));
                        }
                        i++;
                    }
                    jina.setQueryAndDocuments(query, documents);
                    jina.modelReRank();
                    const std::vector<std::pair<float,int>> * rankings = jina.getRankings();
                    response = "";
                    for(size_t i = 0; i < rankings->size(); i++){
                        std::cout<<posToId[(*rankings)[i].second] << " " << std::to_string((*rankings)[i].first) << " " << std::to_string((*rankings)[i].second) << std::endl;
                        if((*rankings)[i].first > 0){
                            response += std::to_string(posToId[(*rankings)[i].second]) + " "; //append in order of most similar first
                        }
                    }
                    std::cout << response <<std::endl;
                    response += "$EOM$";
                    asio::write(socket,asio::buffer(response), ignoredError);
                }
                else{
                    response += " received message that didn't contain a txt file: " + message;
                    asio::write(socket, asio::buffer(response), ignoredError);
                }

            } catch (std::exception const& ex){
                std::cerr << "Error in client loop: " << ex.what() << "\nDisconnecting and returning to listening state" << std::endl;
                break;
            }
        }
    }
    return 0; 
}