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
#include <map>
#include <vector>
#include <set>
#include <functional>
#include <algorithm>
#include <random>
#include <numeric>

#define GLAD_GLX 0
extern "C" {
    #include <glad/glad.h>
}

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include "BPE.h"
#include "Logger.h"

int main(int argc, char** argv) {
    srand(time(nullptr));
 
    NNGL::Logger::getInstance().setLogLevel(NNGL::LogLevel::LL_INFO);
    NNGL::Logger::getInstance().setEnabled(true);

    // Initialize GLFW
    if (!glfwInit()) { LOG_ERROR("GLFW initialization failed!"); return -1; }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

    GLFWwindow* window = glfwCreateWindow(1, 1, "NN Compute", nullptr, nullptr);
    if (!window) {
        LOG_ERROR("GLFW window creation failed!");
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) { LOG_ERROR("Failed to initialize GLAD!"); return -1; }

    std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;

    std::string bpe_file = "bpe50k_v2.checkpoint";
 
    std::vector<std::string> filenames = { "english3.txt", "pg76287.txt", "pg51161.txt", "english3.txt", "pg76287.txt", "pg51161.txt", "english3.txt", "pg76287.txt", "pg51161.txt" };
    std::shared_ptr<NNGL::BPE> bpe = std::make_shared<NNGL::BPE>(1024 * 10);


    // Ensure all printable ASCII single-character tokens are in the vocab
    for (char c = 32; c < 127; ++c) { // printable ASCII
        std::string s(1, c);
        bpe->addToken(s);
    }
    bpe->addToken(" ");
    bpe->trainFromFiles(filenames, true);

    bpe->reduceVocab(50000);
    bpe->addToken("<EOS>");
    bpe->addToken("<PAD>");
    bpe->addToken("<SOS>");

    std::cout << "BPE vocabulary size: " << bpe->getVocabSize() << std::endl;
    bpe->save(bpe_file);
    

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}