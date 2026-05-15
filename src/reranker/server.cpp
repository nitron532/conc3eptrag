#include "reranker.h"
#include <asio.hpp>
#include <fstream>

template<typename Socket>
void clientLoop(Socket& socket, ReRanker& jina, asio::streambuf& readbuffer, std::error_code& ignoredError) {
    std::string_view delim = "$EOM$";
    while(true) {
        try {
            std::string response = "";
            size_t readNum = asio::read_until(socket, readbuffer, delim);
            std::string message = std::string(
                asio::buffers_begin(readbuffer.data()),
                asio::buffers_begin(readbuffer.data()) + readNum
            );
            readbuffer.consume(readNum);
            message = message.substr(0, message.size() - 5);

            std::vector<std::string> documents;
            std::string query;
            std::vector<size_t> posToId;
            std::string line = "";

/*
        while(newline != -1):
            colon = conceptResponse.find(":")
            newline = conceptResponse.find("\n")
            if newline == -1: 
                conceptMapConcepts.append(conceptResponse[colon+1:])
            else: conceptMapConcepts.append(conceptResponse[colon+1:newline])

            conceptResponse = conceptResponse[newline+1:]
*/


            size_t newLine = 0;
            while(newLine != std::string::npos) {
                newLine = message.find("\n");
                line = message.substr(0, newLine);
                if(!isdigit(line[0])) {
                    query = line;
                } else {
                    size_t contextPos = line.find_first_not_of("0123456789");
                    documents.push_back(line.substr(contextPos + 1));
                    posToId.push_back(stoi(line.substr(0, contextPos)));
                }
                message = message.substr(newLine+1);
            }

            jina.setQueryAndDocuments(query, documents);
            jina.modelReRank();
            const std::vector<std::pair<float,int>>* rankings = jina.getRankings();

            response = "";
            for(size_t i = 0; i < rankings->size(); i++) {
                response += std::to_string(posToId[(*rankings)[i].second]) + " "
                            + std::to_string((*rankings)[i].first) + " "
                            + std::to_string((*rankings)[i].second) + "\n";
            }
            std::cout << "response:\n" << response << std::endl;
            response += "$EOM$";
            asio::write(socket, asio::buffer(response), ignoredError);

        } catch(std::exception const& ex) {
            std::cerr << "Error in client loop: " << ex.what()
                      << "\nDisconnecting and returning to listening state" << std::endl;
            break;
        }
    }
}

template<typename Acceptor, typename Socket>
void listenLoop(Acceptor& acceptor, ReRanker& jina, asio::io_context& ioContext,
                asio::streambuf& readbuffer, std::error_code& ignoredError) {
    while(true) {
        Socket socket(ioContext);
        acceptor.accept(socket);
        std::cout << "accepted a client" << std::endl;
        clientLoop(socket, jina, readbuffer, ignoredError);
    }
}

int main(int argc, char* argv[]) {
    int useTCP = -1;
    int portNum = 0;
    const char* socketPath = nullptr;

    try {
        useTCP = atoi(argv[1]);
        if(useTCP) portNum = atoi(argv[2]);
        else        socketPath = argv[2];
    } catch(std::exception const& ex) {
        std::cerr << "Usage: ./rerankerserver <tcp(1) or uds(0)> <connectionPoint>" << std::endl;
        exit(1);
    }

    ReRanker jina(0, "jina-reranker-v3-Q8_0.gguf");
    asio::io_context ioContext;
    std::error_code ignoredError;
    asio::streambuf readbuffer;

    if(useTCP) {
        asio::ip::tcp::acceptor acceptor(
            ioContext,
            asio::ip::tcp::endpoint(asio::ip::tcp::v4(), portNum)
        );
        std::cout << "Listening on TCP port: " << portNum << std::endl;
        listenLoop<asio::ip::tcp::acceptor, asio::ip::tcp::socket>(
            acceptor, jina, ioContext, readbuffer, ignoredError
        );
    } else {
        ::unlink(socketPath);
        asio::local::stream_protocol::endpoint ep(socketPath);
        asio::local::stream_protocol::acceptor acceptor(ioContext, ep);
        std::cout << "Listening on UDS: " << socketPath << std::endl;
        listenLoop<asio::local::stream_protocol::acceptor, asio::local::stream_protocol::socket>(
            acceptor, jina, ioContext, readbuffer, ignoredError
        );
    }

    return 0;
}