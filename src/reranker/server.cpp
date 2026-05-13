#include "reranker.h"
#include <asio.hpp>
#include <fstream>

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

    //TODO use unix domain socket for local comms. keep an option for TCP though, if somebody wants to run reranker on a different computer
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
        while(true){ //single client loop
            try{
                std::string response = "Server at " + socket.local_endpoint().address().to_string() + ":" + std::to_string(socket.local_endpoint().port());
                readNum = asio::read_until(socket, readbuffer, delim);
                message = std::string(asio::buffers_begin(readbuffer.data()),asio::buffers_begin(readbuffer.data())+readNum);
                readbuffer.consume(readNum);
                // std::cout << "received: " << message << std::endl;
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
                        response += std::to_string(posToId[(*rankings)[i].second]) + " " + std::to_string((*rankings)[i].first) + " " + std::to_string((*rankings)[i].second) + "\n";
                    }
                    std::cout <<"response: \n" << response <<std::endl;
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