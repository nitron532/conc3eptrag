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