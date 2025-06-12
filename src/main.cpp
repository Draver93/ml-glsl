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


// Batch processing for better GPU utilization
const int BATCH_SIZE = 64;
const int STEPS = 10000000; // Reduced for testing

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

void generateBatchData(std::vector<float>& batch_inputs, std::vector<float>& batch_targets) {
    for (int i = 0; i < BATCH_SIZE; i++) {
        float a = 3.14f * ((float)rand() / RAND_MAX) * 2.0f - 3.14f;
        float b_val = 3.14f * ((float)rand() / RAND_MAX) * 2.0f - 3.14f;

        batch_inputs[i * 2 + 0] = a / 3.14f;
        batch_inputs[i * 2 + 1] = b_val / 3.14f;
        batch_targets[i] = std::sin(a) * std::sin(b_val);
    }
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
    std::cout << "Using batch size: " << BATCH_SIZE << std::endl;


    // Network setup - simplified for testing
    const int input_size = 2;
    const int hidden_size = 15;
    const int output_size = 1;

    NNGL::NeuralNetwork nn;
    
    nn.addLayer(std::make_unique<NNGL::Layer>(input_size, hidden_size, BATCH_SIZE, NNGL::ActivationFnType::TANH));
    nn.addLayer(std::make_unique<NNGL::Layer>(hidden_size, hidden_size, BATCH_SIZE, NNGL::ActivationFnType::RELU));
    nn.addLayer(std::make_unique<NNGL::Layer>(hidden_size, hidden_size, BATCH_SIZE, NNGL::ActivationFnType::LRELU));
    nn.addLayer(std::make_unique<NNGL::Layer>(hidden_size, hidden_size, BATCH_SIZE, NNGL::ActivationFnType::SIGMOID));
    nn.addLayer(std::make_unique<NNGL::Layer>(hidden_size, hidden_size, BATCH_SIZE, NNGL::ActivationFnType::TANH));
    nn.addLayer(std::make_unique<NNGL::Layer>(hidden_size, output_size, BATCH_SIZE, NNGL::ActivationFnType::IDENTITY));

    // Training data
    std::vector<float> batch_inputs(BATCH_SIZE * input_size);
    std::vector<float> batch_targets(BATCH_SIZE * output_size);

    float learning_rate = 0.03f;
    int steps_left = STEPS / BATCH_SIZE;

    // Main training loop
    while (steps_left > 0) {
        // Generate batch data with size validation
        generateBatchData(batch_inputs, batch_targets);
        // Upload batch to GPU with size validation
        nn.train(batch_inputs, batch_targets, learning_rate);

        // Periodic GPU synchronization and cleanup
        if (steps_left % 100 == 0) {
            glFinish(); // Force GPU to complete all pending operations
            // Check for OpenGL errors
            GLenum error = glGetError();
            if (error != GL_NO_ERROR) throw std::runtime_error("OpenGL Error detected: " + std::to_string(error) + " at training step " + std::to_string(STEPS / BATCH_SIZE - steps_left));
        }

        // Status updates and weight display
        if (0 && steps_left % (10000 / BATCH_SIZE) == 0) {
            resetCursor();
            // Memory debugging info
            std::cout << "Batch sizes - Input: " << batch_inputs.size() << ", Target: " << batch_targets.size() << std::endl;
            for (int i = 0; i < nn.m_Layers.size(); i++) {
                nn.m_Layers[i]->printHeatmap();
                std::cout << std::endl;
            }

            std::cout << "\nStep: " << (STEPS / BATCH_SIZE - steps_left) << " LR: " << learning_rate << std::endl;
        }

        // Learning rate decay
        if (steps_left % 10000 == 0) {
            learning_rate *= 0.95f;
        }

        steps_left--;
    }

    std::cout << "\nTraining completed!" << std::endl;

    nn.run();

    std::cout << "Goodbye!" << std::endl;

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}