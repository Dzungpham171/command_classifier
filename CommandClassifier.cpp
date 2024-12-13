#include "CommandClassifier.h"
#define MLPACK_ENABLE_ANN_SERIALIZATION


// Load GloVe embeddings from file
void CommandClassifier::loadGloveEmbeddings(const std::string& filePath) {
    std::ifstream file(filePath.c_str());
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string word;
        iss >> word;
        std::vector<double> embedding(embeddingDim);
        for (int i = 0; i < embeddingDim; ++i) {
            iss >> embedding[i];
        }
        gloveEmbeddings[word] = embedding;
    }
}

// Helper function to tokenize text
std::vector<std::string> CommandClassifier::tokenize(const std::string& text) {
    std::istringstream stream(text);
    std::vector<std::string> tokens;
    std::string token;
    while (stream >> token) {
        tokens.push_back(token);
    }
    return tokens;
}

// Vectorize a command using GloVe embeddings
std::vector<double> CommandClassifier::vectorizeCommand(const std::string& command) {
    std::vector<double> vector(embeddingDim, 0.0);
    auto tokens = tokenize(command);
    for (const auto& token : tokens) {
        if (gloveEmbeddings.find(token) != gloveEmbeddings.end()) {
            const auto& embedding = gloveEmbeddings[token];
            for (int i = 0; i < embeddingDim; ++i) {
                vector[i] += embedding[i];
            }
        }
    }
    return vector;
}

void CommandClassifier::train(const std::vector<std::string>& commands, const std::vector<int>& labels) {
    arma::mat trainData(embeddingDim, commands.size());
    arma::mat trainLabels(numCommands, commands.size(), arma::fill::zeros);

    for (size_t i = 0; i < commands.size(); ++i) {
        std::vector<double> embedding = vectorizeCommand(commands[i]);
        for (size_t j = 0; j < embeddingDim; ++j) {
            trainData(j, i) = embedding[j];
        }
        trainLabels(labels[i], i) = 1.0;
    }

    // Define the model architecture
    model.Add<mlpack::Linear>(64);
    model.Add<mlpack::ReLU>();
    model.Add<mlpack::Linear>(128);
    model.Add<mlpack::ReLU>();
	model.Add<mlpack::Linear>(64);
	model.Add<mlpack::ReLU>();
	model.Add<mlpack::Linear>(numCommands);
    model.Add<mlpack::LogSoftMax>();

    // Set up the optimizer
    ens::Adam optimizer(0.001, 64, 0.9, 0.999, 1e-8, 1000000, 1e-8, true);

    // Train the model
    model.Train(trainData, trainLabels, optimizer);

    std::cout << "Model trained successfully." << std::endl;
}

// Save the model to a file
void CommandClassifier::saveModel(const std::string& modelFile) {
    mlpack::data::Save(modelFile, "model", model, false);
    std::cout << "Model saved to " << modelFile << std::endl;
}

// Load the model from a file
void CommandClassifier::loadModel(const std::string& modelFile) {
    mlpack::data::Load(modelFile, "model", model, false);
    std::cout << "Model loaded from " << modelFile << std::endl;
}

// Classify a new command
std::string CommandClassifier::classify(const std::string& command) {
    auto vec = vectorizeCommand(command);
    arma::mat input(embeddingDim, 1);
    for (int i = 0; i < embeddingDim; ++i) {
        input(i, 0) = vec[i];
    }

    arma::mat output;
    model.Predict(input, output);
    size_t predictedClass = output.index_max();
    return commandMapping[predictedClass];
}