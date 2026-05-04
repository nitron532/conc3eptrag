from socket import *
serverName = '127.0.0.1'
serverPort = 2020
clientSocket = socket(AF_INET, SOCK_STREAM)
clientSocket.connect((serverName, serverPort))
sentence = input("input test query:\n")
while(sentence != '@'):
    clientSocket.send(sentence.encode())
    serverResponse = clientSocket.recv(1024)
    print("From server:", serverResponse.decode())
    sentence = input("input test query:\n")
clientSocket.close()