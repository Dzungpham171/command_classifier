#pragma once
#define MLPACK_ENABLE_ANN_SERIALIZATION
#include <mlpack.hpp>
#include <mlpack/methods/ann/ann.hpp>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <sstream>
#include <string>
#include <iostream>
#include <cstring>

class CommandClassifier {
private:
    // GloVe Embeddings
    std::unordered_map<std::string, std::vector<double>> gloveEmbeddings;

    // Mapping command indices to valid command names
    std::vector<std::string> commandMapping = {
        "invalid command",
        "shutdown", "restart",
        "list app", "start app", "stop app",
        "list service", "start service", "stop service",
        "get file", "delete file",
        "screenshot",
        "start webcam", "stop webcam",
        "start keylogger", "stop keylogger"
    };
    const int embeddingDim = 50; // GloVe dimension
    const int numCommands = commandMapping.size(); // Number of valid commands
	mlpack::ann::FFN<mlpack::ann::KLDivergence, mlpack::ann::RandomInitialization> model;

    // Helper function to tokenize text
    std::vector<std::string> tokenize(const std::string& text);

    // Vectorize a command using GloVe embeddings
    std::vector<double> vectorizeCommand(const std::string& command);

public:
    // Load GloVe embeddings from file
    void loadGloveEmbeddings(const std::string& filePath);

    // Train the model
    void train(const std::vector<std::string>& commands, const std::vector<int>& labels);

    // Save the model to a file
    void saveModel(const std::string& modelFile);

    // Load the model from a file
    void loadModel(const std::string& modelFile);

    // Classify a new command
    std::string classify(const std::string& command);
};
