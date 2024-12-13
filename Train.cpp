#define MLPACK_ENABLE_ANN_SERIALIZATION

#include "CommandClassifier.h"
#include <iostream>
#include <vector>
#include <string>

static void getDataTraining(std::vector<std::string>& commands, std::vector<int>& labels)
{
    using namespace std;
    ifstream in("command_data_training.csv");
    string line;
    getline(in, line);
    while (getline(in, line))
    {
        string command;
        int label;
        stringstream ss(line);
        getline(ss, command, ',');
        ss >> label;
        commands.push_back(command);
        labels.push_back(label);
    }
    in.close();
}

int main() {
    // Create an instance of CommandClassifier
    CommandClassifier classifier;

    // Load GloVe embeddings from a file
    std::string gloveFilePath = "extract_words.txt";
    classifier.loadGloveEmbeddings(gloveFilePath);

    // Sample commands and labels for training
    std::vector<std::string> commands;
    std::vector<int> labels; // Example labels for the commands
    getDataTraining(commands, labels);

    // Train the model
    classifier.train(commands, labels);

    // Save the model to a file
    std::string modelFilePath = "model.json";
    classifier.saveModel(modelFilePath);
	std::cout << "Successfully save the model to " << modelFilePath << std::endl;
    classifier.loadModel("model.json");
    while (true)
    {
        std::string command;
        std::cout << "Enter a command: ";
        std::getline(std::cin, command);
        if (command == "0") break;
        std::string real_command = classifier.classify(command);
        std::cout << "Real command: " << real_command << std::endl;
    }
	system("pause");
    return 0;
}
