#include "reranker.h"
#include <asio.hpp>
#include <asio/ssl.hpp>
#include <openssl/ssl.h>
#include <fstream>
#include <type_traits>

template<typename Socket>
void clientLoop(Socket& socket, ReRanker& jina, std::error_code& ignoredError) {
    std::string_view delim = "$EOM$";
    asio::streambuf readbuffer;
    while(true) {
        try {
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

            std::string response = "";
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

template<typename AcceptorFunction>
void listenLoop(AcceptorFunction&& acceptClient, ReRanker& jina){
    while(true){
        try{
            auto sock = acceptClient();
            clientLoop(sock, jina);
        } catch(const std::exception& e) {
            std::cerr << "Failed to accept client: " << e.what() << std::endl;
        }
    }
}

struct Config {
    std::string certPath = "/etc/conc3ept/tls/server.pem";
    std::string keyPath  = "/etc/conc3ept/tls/server-key.pem";
    int method = 0; // 1 for TCP 0 for UDS
    int port = 33333;
    std::string sockPath = "/tmp/conc3ept";
};

Config parseArgs(int argc, char* argv[]) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];

        if (a == "--cert" && i + 1 < argc) cfg.certPath = argv[++i];
        else if (a == "--key" && i + 1 < argc) cfg.keyPath = argv[++i];
        else if (a == "--method" && i +1 < argc) cfg.method = atoi(argv[++i]);
        else if (a == "--sockPath" && i + 1 < argc) cfg.socketPath = argv[++i];
        else if (a == "--port" && i + 1 < argc) cfg.port = atoi(argv[++i]);
    }
    if (const char* e = std::getenv("TLS_CERT")) cfg.certPath = e;
    if (const char* e = std::getenv("TLS_KEY"))  cfg.keyPath  = e;
    return cfg;
}

int main(int argc, char* argv[]) {
    if(argc < 3){
        std::cerr << "Usage: ./rerankerserver <tcp(1) or uds(0)>" << std::endl;
        std::cerr<< "For TCP/TLS, add --cert <path to cert> --key <path to key>" << std::endl;
        std::cerr<< "For UDS, add --sockPath <socket path>" << std::endl;
        exit(1);
    }
    Config cfg = parseArgs(argc, argv);

    ReRanker jina(0, "models/jina-reranker-v3-Q8_0.gguf");
    asio::io_context ioContext;
    std::error_code ignoredError;

    if(cfg.method) {
        asio::ip::tcp::acceptor acceptor(
            ioContext,
            asio::ip::tcp::endpoint(asio::ip::tcp::v4(), cfg.port)
        );
        std::cout << "Listening on TCP port: " << cfg.port << std::endl;
        asio::ssl::context ctx(asio::ssl::context::tls_server);
        ctx.set_options(asio::ssl::context::default_workarounds);
        SSL_CTX_set_min_proto_version(ctx.native_handle(), TLS1_2_VERSION);
        ctx.use_private_key_file(cfg.keyPath, asio::ssl::context::pem); 
        ctx.use_certificate_chain_file(cfg.certPath);
        listenLoop([&]{
            asio::ssl::stream<asio::ip::tcp::socket> sock(ioContext,ctx);
            acceptor.accept(sock.lowest_layer());
            sock.handshake(asio::ssl::stream_base::server);
            return sock;
        }, jina);
    } else {
        ::unlink(cfg.sockPath);
        asio::local::stream_protocol::endpoint ep(cfg.sockPath);
        asio::local::stream_protocol::acceptor acceptor(ioContext, ep);
        std::cout << "Listening on UDS: " << socketPath << std::endl;
        listenLoop([&]{
            asio::local::stream_protocol::socket sock(ioContext);
            acceptor.accept(sock);
            return sock;
        }, jina);
    }
    return 0;
}