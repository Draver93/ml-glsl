#include "Layer.h"

#include <execution>
#include <string>
#include <iostream>

namespace NNGL {
	Layer::Layer(int width, int height, int batchSize, ActivationFnType type) {
        m_Width = width;
        m_Height = height;
        m_ActivationFnType = type;

        // Initialize weights
        std::vector<float> weights(m_Width * m_Height);
        for (auto& w : weights) w = NNGL::activationFunctions[type].weight_initializer(m_Width, m_Height);

        glGenBuffers(1, &m_WeightBuffer);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_WeightBuffer);
        glBufferData(GL_SHADER_STORAGE_BUFFER, weights.size() * sizeof(float), weights.data(), GL_DYNAMIC_DRAW);

        // for ADAM calc
        glGenBuffers(1, &m_ADAM_MBuffer);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ADAM_MBuffer);
        glBufferData(GL_SHADER_STORAGE_BUFFER, weights.size() * sizeof(float), nullptr, GL_DYNAMIC_DRAW);

        glGenBuffers(1, &m_ADAM_VBuffer);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ADAM_VBuffer);
        glBufferData(GL_SHADER_STORAGE_BUFFER, weights.size() * sizeof(float), nullptr, GL_DYNAMIC_DRAW);

        // Initialize biases
        std::vector<float> biases(m_Height);
        for (auto& b : biases) b = NNGL::activationFunctions[type].weight_initializer(m_Width, m_Height);

        glGenBuffers(1, &m_BiasBuffer);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_BiasBuffer);
        glBufferData(GL_SHADER_STORAGE_BUFFER, biases.size() * sizeof(float), biases.data(), GL_DYNAMIC_DRAW);

        // Initialize activation and delta buffers
        glGenBuffers(1, &m_ActivationBuffer);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ActivationBuffer);
        glBufferData(GL_SHADER_STORAGE_BUFFER, batchSize * m_Height * sizeof(float), nullptr, GL_DYNAMIC_DRAW);

        glGenBuffers(1, &m_PreactivationBuffer);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_PreactivationBuffer);
        glBufferData(GL_SHADER_STORAGE_BUFFER, batchSize * m_Height * sizeof(float), nullptr, GL_DYNAMIC_DRAW);

        glGenBuffers(1, &m_DeltaBuffer);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_DeltaBuffer);
        glBufferData(GL_SHADER_STORAGE_BUFFER, batchSize * m_Height * sizeof(float), nullptr, GL_DYNAMIC_DRAW);

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	}

	Layer::~Layer() {

        if(m_WeightBuffer)          glDeleteBuffers(1, &m_WeightBuffer);
        if(m_BiasBuffer)            glDeleteBuffers(1, &m_BiasBuffer);
        if(m_ActivationBuffer)      glDeleteBuffers(1, &m_ActivationBuffer);
        if(m_PreactivationBuffer)   glDeleteBuffers(1, &m_PreactivationBuffer);
        if(m_DeltaBuffer)           glDeleteBuffers(1, &m_DeltaBuffer);
	}

    const std::string colors[] = {
    "\033[48;5;17m",   // deep blue
    "\033[48;5;18m",
    "\033[48;5;19m",
    "\033[48;5;20m",
    "\033[48;5;21m",   // blue
    "\033[48;5;38m",   // teal
    "\033[48;5;44m",   // cyan
    "\033[48;5;51m",   // light cyan
    "\033[48;5;87m",   // white-blue
    "\033[48;5;123m",  // white-cyan
    "\033[48;5;159m",  // white
    "\033[48;5;190m",  // light yellow
    "\033[48;5;226m",  // yellow
    "\033[48;5;220m",  // orange
    "\033[48;5;202m",  // orange-red
    "\033[48;5;196m",  // bright red
    "\033[0m"          // reset
    };

    void Layer::printHeatmap() {
        std::vector<float> weights(m_Width * m_Height);

        // Read weights
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_WeightBuffer);
        float* weightPtr = (float*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
        std::copy(weightPtr, weightPtr + weights.size(), weights.begin());
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

        // Find min/max for normalization
        auto [minIt, maxIt] = std::minmax_element(weights.begin(), weights.end());
        float minVal = *minIt, maxVal = *maxIt;

        for (int i = 0; i < m_Width; i++) {
            for (int j = 0; j < m_Height; j++) {
                // Normalize value between 0 and 1
                float normalized = (weights[i * m_Height + j] - minVal) / (maxVal - minVal);
                int colorIdx = static_cast<int>(normalized * (sizeof(colors) / sizeof(colors[0]) - 2));
                std::cout << colors[colorIdx] << "  " << colors[16]; // print 2 spaces with bg color, then reset
            }
            std::cout << "\n";
        }

        std::cout << "\033[0m\n";
        std::cout << std::endl;
    }

    void Layer::displayLayer(const std::string& layerName) {
        std::cout << "\n=== " << layerName << " ===" << std::endl;

        // Read weights
        std::vector<float> weights(m_Width * m_Height);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_WeightBuffer);
        float* weightPtr = (float*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
        std::copy(weightPtr, weightPtr + weights.size(), weights.begin());
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

        // Display weight matrix
        std::cout << "Weights (" << m_Width << "x" << m_Height << "):" << std::endl;
        for (int i = 0; i < m_Width; i++) {
            for (int j = 0; j < m_Height; j++) {
                printf("%8.4f ", weights[i * m_Height + j]);
            }
            std::cout << std::endl;
        }

        // Read and display biases
        std::vector<float> biases(m_Height);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER,m_BiasBuffer);
        float* biasPtr = (float*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
        std::copy(biasPtr, biasPtr + biases.size(), biases.begin());
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

        std::cout << "Biases: ";
        for (float b : biases) {
            printf("%8.4f ", b);
        }
        std::cout << std::endl;
    }
}