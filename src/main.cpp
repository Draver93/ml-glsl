#define NOMINMAX
#include <windows.h>
#include <chrono>
#include  <sstream>
#include <string>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <array>
#include <vector>
#include <functional>
#include <algorithm>
#include <random>

#include <glm/glm.hpp>

#include "ActivationFunctions.h"
#include "NeuralNetwork.h"
#include "MNISTLoader.h"



// Random float between -1 and 1
float random_norm_val() {
    return ((float)rand() / RAND_MAX) * 2 - 1;
}

void resetCursor() {
#ifdef _WIN32
    // Windows console API (best way)
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    COORD coord = { 0, 0 };
    SetConsoleCursorPosition(hConsole, coord);
#else
    // ANSI escape sequence for UNIX-like systems
    std::cout << "\033[H"; // Move cursor to top-left
#endif
}

void digit_recognition() {
    // Network setup - simplified for testing
    const int inputSize = 784;
    const int hiddenSize = 20;
    const int outputSize = 10;

    const int batchSize = 8;
    const int steps = 5000000;

    NNGL::NeuralNetwork nn(batchSize);


    std::vector<std::vector<uint8_t>> trainImages = NNGL::MNIST::loadImages("mnist/train-images.idx3-ubyte");
    std::vector<uint8_t> trainLabels = NNGL::MNIST::loadLabels("mnist/train-labels.idx1-ubyte");

    // Load test data
    std::vector<std::vector<uint8_t>> testImages = NNGL::MNIST::loadImages("mnist/t10k-images.idx3-ubyte");
    std::vector<uint8_t> testLabels = NNGL::MNIST::loadLabels("mnist/t10k-labels.idx1-ubyte");

    // Track current position in dataset
    // Separate indices for train and test
    static size_t trainIndex = 0;
    static size_t testIndex = 0;

    const size_t totalTrainSamples = trainImages.size();
    const size_t totalTestSamples = testImages.size();

    nn.onTrainBatch([&](std::vector<float>& batchInputs, std::vector<float>& batchTargets, int batchSize) {
        for (int i = 0; i < batchSize; ++i) {
            // Use training data with wraparound
            size_t sampleIndex = (trainIndex + i) % totalTrainSamples;

            // Convert image from uint8_t to float and normalize (0-255 → 0-1)
            const auto& image = trainImages[sampleIndex];
            size_t inputOffset = i * 784; // 28*28 = 784 pixels per image
            for (size_t pixel = 0; pixel < 784; ++pixel) {
                batchInputs[inputOffset + pixel] = static_cast<float>(image[pixel]) / 255.0f;
            }

            // Convert label to one-hot encoding
            uint8_t label = trainLabels[sampleIndex];
            size_t targetOffset = i * 10; // 10 classes (digits 0-9)

            // Clear all classes first
            for (size_t j = 0; j < 10; ++j) {
                batchTargets[targetOffset + j] = 0.0f;
            }
            // Set the correct class to 1.0
            batchTargets[targetOffset + label] = 1.0f;
        }
        // Update train index for next batch
        trainIndex = (trainIndex + batchSize) % totalTrainSamples;
    });

    nn.onTestBatch([&](std::vector<float>& batchInputs, std::vector<float>& batchTargets, int batchSize) {
        for (int i = 0; i < batchSize; ++i) {
            // Use test data with wraparound
            size_t sampleIndex = (testIndex + i) % totalTestSamples;

            // Convert image from uint8_t to float and normalize (0-255 → 0-1)
            const auto& image = testImages[sampleIndex];
            size_t inputOffset = i * 784; // 28*28 = 784 pixels per image
            for (size_t pixel = 0; pixel < 784; ++pixel) {
                batchInputs[inputOffset + pixel] = static_cast<float>(image[pixel]) / 255.0f;
            }

            // Convert label to one-hot encoding
            uint8_t label = testLabels[sampleIndex];
            size_t targetOffset = i * 10; // 10 classes (digits 0-9)

            // Clear all classes first
            for (size_t j = 0; j < 10; ++j) {
                batchTargets[targetOffset + j] = 0.0f;
            }
            // Set the correct class to 1.0
            batchTargets[targetOffset + label] = 1.0f;
        }
        // Update test index for next batch
        testIndex = (testIndex + batchSize) % totalTestSamples;
    });


    nn.addLayer(inputSize, hiddenSize, NNGL::ActivationFnType::RELU);
    nn.addLayer(hiddenSize, hiddenSize, NNGL::ActivationFnType::RELU);
    nn.addLayer(hiddenSize, outputSize, NNGL::ActivationFnType::IDENTITY);

    float learningRate = 0.03f;
    int stepsLeft = steps / batchSize;

    while (stepsLeft > 0) {
        nn.train(learningRate);

        // Periodic GPU synchronization and cleanup
        if (stepsLeft % 100 == 0) {
            glFinish(); // Force GPU to complete all pending operations
            GLenum error = glGetError(); // Check for OpenGL errors
            if (error != GL_NO_ERROR) throw std::runtime_error("OpenGL Error detected: " + std::to_string(error) + " at training step " + std::to_string(steps / batchSize - stepsLeft));
        }

        // Status updates and weight display
        if (stepsLeft % (10000 / batchSize) == 0) {
            resetCursor();
            for (int i = 1; i < nn.m_Layers.size(); i++) {
                nn.m_Layers[i]->printHeatmap();
                std::cout << std::endl;
            }
            std::cout << "\nStep: " << (steps / batchSize - stepsLeft) * batchSize << " LR: " << learningRate << std::endl;
        }

        // Learning rate decay
        if (stepsLeft % 10000 == 0) learningRate *= 0.99f;

        stepsLeft--;
    }

    std::cout << "\nTraining completed!" << std::endl;

    nn.run();
}

void sin_multiplication() {

    // Network setup - simplified for testing
    const int inputSize = 2;
    const int hiddenSize = 20;
    const int outputSize = 1;

    const int batchSize = 64;
    const int steps = 1000000; 

    NNGL::NeuralNetwork nn(batchSize);

    nn.onTrainBatch([&](std::vector<float>& batchInputs, std::vector<float>& batchTargets, int batchSize) {
        for (int i = 0; i < batchSize; i++) {
            float a = 3.14f * ((float)rand() / RAND_MAX) * 2.0f - 3.14f;
            float b_val = 3.14f * ((float)rand() / RAND_MAX) * 2.0f - 3.14f;

            batchInputs[i * 2 + 0] = a / 3.14f;
            batchInputs[i * 2 + 1] = b_val / 3.14f;
            batchTargets[i] = std::sin(a) * std::sin(b_val);
        }    
    });
    nn.onTestBatch([&](std::vector<float>& batchInputs, std::vector<float>& batchTargets, int batchSize) {
        for (int i = 0; i < batchSize; i++) {
            float a = 3.14f * ((float)rand() / RAND_MAX) * 2.0f - 3.14f;
            float b_val = 3.14f * ((float)rand() / RAND_MAX) * 2.0f - 3.14f;

            batchInputs[i * 2 + 0] = a / 3.14f;
            batchInputs[i * 2 + 1] = b_val / 3.14f;
            batchTargets[i] = std::sin(a) * std::sin(b_val);
        }
    });

    nn.addLayer(inputSize, 1, NNGL::ActivationFnType::TANH);
    nn.addLayer(1, hiddenSize, NNGL::ActivationFnType::RELU);
    nn.addLayer(hiddenSize, hiddenSize, NNGL::ActivationFnType::LRELU);
    nn.addLayer(hiddenSize, hiddenSize, NNGL::ActivationFnType::SIGMOID);
    nn.addLayer(hiddenSize, hiddenSize, NNGL::ActivationFnType::TANH);
    nn.addLayer(hiddenSize, outputSize, NNGL::ActivationFnType::IDENTITY);

    float learningRate = 0.03f;
    int stepsLeft = steps / batchSize;

    while (stepsLeft > 0) {
        nn.train(learningRate);

        // Periodic GPU synchronization and cleanup
        if (stepsLeft % 100 == 0) {
            glFinish(); // Force GPU to complete all pending operations
            GLenum error = glGetError(); // Check for OpenGL errors
            if (error != GL_NO_ERROR) throw std::runtime_error("OpenGL Error detected: " + std::to_string(error) + " at training step " + std::to_string(steps / batchSize - stepsLeft));
        }

        // Status updates and weight display
        if (stepsLeft % (10000 / batchSize) == 0) {
            resetCursor();
            for (int i = 0; i < nn.m_Layers.size(); i++) {
                nn.m_Layers[i]->printHeatmap();
                std::cout << std::endl;
            }
            std::cout << "\nStep: " << (steps / batchSize - stepsLeft) * batchSize << " LR: " << learningRate << std::endl;
        }

        // Learning rate decay
        if (stepsLeft % 10000 == 0) learningRate *= 0.99f;

        stepsLeft--;
    }

    std::cout << "\nTraining completed!" << std::endl;

    nn.run();
}

int main() {
    srand(time(nullptr));

    // Initialize GLFW
    if (!glfwInit()) { std::cerr << "GLFW initialization failed!" << std::endl; return -1; }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

    GLFWwindow* window = glfwCreateWindow(1, 1, "NN Compute", nullptr, nullptr);
    if (!window) {
        std::cerr << "GLFW window creation failed!" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) { std::cerr << "Failed to initialize GLAD!" << std::endl; return -1; }

    std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;

    digit_recognition();

    std::cout << "Goodbye!" << std::endl;

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}