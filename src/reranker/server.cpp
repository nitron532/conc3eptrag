#include "reranker.h"
#include <asio.hpp>

using asio::ip::tcp;

int main(int argc, char *argv[])
{
    if (argc < 2) {
        std::cerr << "Usage: ./rerankerserver <portnumber>" << std::endl;
        exit(1);
    }
    int portNum = atoi(argv[1]);

    //initialize llama.cpp reranker
    ReRanker jina(0,"jina-reranker-v3-Q8_0.gguf");
    
    std::string q = "What are the health benefits of green tea?";
    std::vector<std::string> documents = {
    "Green tea contains antioxidants called catechins that may help reduce inflammation and protect cells from damage.",
    "El precio del café ha aumentado un 20% este año debido a problemas en la cadena de suministro.",
    "Studies show that drinking green tea regularly can improve brain function and boost metabolism.",
    "Basketball is one of the most popular sports in the United States.",
    "绿茶富含儿茶素等抗氧化剂，可以降低心脏病风险，还有助于控制体重。",
    "Le thé vert est riche en antioxydants et peut améliorer la fonction cérébrale.",
    };

    jina.setQueryAndDocuments(q, documents);
    jina.modelReRank();
    const std::vector<std::pair<float,int>> * rankings = jina.getRankings();

    for(int i = 0; i < rankings->size(); i++){
        std::cout << std::to_string((*rankings)[i].first) << " " << std::to_string((*rankings)[i].second) << std::endl;
    }




    return 0;

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
        std::string response = "ok " + std::to_string(clientPort) + " I see you";
        while(true){ //single client loop
            try{
                readNum = asio::read_until(socket, readbuffer, delim);
                message = std::string(asio::buffers_begin(readbuffer.data()),asio::buffers_begin(readbuffer.data())+readNum);
                readbuffer.consume(readNum);
                std::cout << "received: " << message << std::endl;

                asio::write(socket, asio::buffer(response), ignoredError);
            } catch (std::exception const& ex){
                std::cerr << "Error in client loop: " << ex.what() << "\nDisconnecting and returning to listening state" << std::endl;
                break;
            }
        }
    }
    return 0; 
}