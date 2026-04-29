/* A simple server in the internet domain using TCP
   The port number is passed as an argument */
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h> 
#include <sys/socket.h>
#include <netinet/in.h>
#include <llama-cpp.h>
#include <string>

void error(const char *msg)
{
    perror(msg);
    exit(1);
}

int main(int argc, char *argv[])
{

    //initialize llama.cpp
    llama_backend_init();
    auto modelParams = llama_model_default_params();
    modelParams.n_gpu_layers = 0;

    const char* rerankingModelPath = "jina-reranker-v3-Q8_0.gguf";

    llama_model* rerankingModel = llama_model_load_from_file(
        rerankingModelPath,
        modelParams
    );

    if(!rerankingModel){
        error("Couldn't find model at specified path.\n");
    }
    
    llama_context_params contextParams = llama_context_default_params();
    llama_context* context = llama_init_from_model(rerankingModel, contextParams);

    int sockfd, newsockfd, portno;
    socklen_t clilen;
    char buffer[256];
    struct sockaddr_in serv_addr, cli_addr;
    int n;
    if (argc < 2) {
        fprintf(stderr,"ERROR, no port provided\n");
        exit(1);
    }
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) 
        error("ERROR opening socket");
    bzero((char *) &serv_addr, sizeof(serv_addr));
    portno = atoi(argv[1]);
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = INADDR_ANY;
    serv_addr.sin_port = htons(portno);
    if (bind(sockfd, (struct sockaddr *) &serv_addr,
            sizeof(serv_addr)) < 0) 
            error("ERROR on binding");
    std::cout << "Listening on " << serv_addr.sin_addr.s_addr << " " << serv_addr.sin_port << std::endl;
    listen(sockfd,5);
    clilen = sizeof(cli_addr);
    newsockfd = accept(sockfd, 
                (struct sockaddr *) &cli_addr, 
                &clilen);
    if (newsockfd < 0) 
        error("ERROR on accept");
    
    while(true){ //only accept connection from one client. maybe add option to handle multiple connections
        bzero(buffer,256);
        while(recv(newsockfd, buffer, 255,0) > 0){
            printf("message so far: %s\n",buffer);
            write(newsockfd,"I got your message",18);

        }
        if (n < 0) error("ERROR reading from socket");
        printf("Here is the message: %s\n",buffer);
        n = write(newsockfd,"I got your message",18);
        if (n < 0) error("ERROR writing to socket");
    }

    //Deallocate resources
    close(newsockfd);
    close(sockfd);
    llama_free(context);
    llama_model_free(rerankingModel);
    llama_backend_free();
    return 0; 
}