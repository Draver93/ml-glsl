#include "Layer.h"
#include "Logger.h"

#include <execution>
#include <string>
#include <iostream>

namespace NNGL {
	Layer::Layer(int width, int height, int batchSize, ActivationFnType type) 
        : m_Width(width), m_Height(height), m_ActivationFnType(type) {

        LOG_DEBUG("[LAYER INIT] Creating layer " + std::to_string(m_Width) + "x" + std::to_string(m_Height) + " with batch size " + std::to_string(m_ActivationFnType));

        // Initialize biases
        m_WeightMat = std::make_shared<Matrix>(m_Width, m_Height);
        for (int i = 0; i < m_WeightMat->rows; i++)
            for (int j = 0; j < m_WeightMat->cols; j++)
                m_WeightMat->set(i, j, NNGL::activationFunctions[m_ActivationFnType].weight_initializer(m_Width, m_Height));
        m_WeightMat->uploadToGPU();
        LOG_DEBUG("[GPU BUFFER] Created weight buffer " + std::to_string(m_WeightMat->buffer) + " (" + std::to_string(m_Width * m_Height * sizeof(float)) + " bytes)");


        // for ADAM calc
        m_ADAM_M_Mat = std::make_shared<Matrix>(m_Width, m_Height);
        m_ADAM_M_Mat->uploadToGPU();
        LOG_DEBUG("[GPU BUFFER] Created ADAM M buffer " + std::to_string(m_ADAM_M_Mat->buffer) + " (" + std::to_string(m_Width * m_Height * sizeof(float)) + " bytes)");

        m_ADAM_V_Mat = std::make_shared<Matrix>(m_Width, m_Height);
        m_ADAM_V_Mat->uploadToGPU();
        LOG_DEBUG("[GPU BUFFER] Created ADAM V buffer " + std::to_string(m_ADAM_V_Mat->buffer) + " (" + std::to_string(m_Width * m_Height * sizeof(float)) + " bytes)");

        // Initialize biases
        m_BiasMat = std::make_shared<Matrix>(m_Height, 1);
        for (int i = 0; i < m_BiasMat->rows; i++) m_BiasMat->set(i, 0, NNGL::activationFunctions[m_ActivationFnType].weight_initializer(m_Width, m_Height));
        m_BiasMat->uploadToGPU();
        LOG_DEBUG("[GPU BUFFER] Created bias buffer " + std::to_string(m_BiasMat->buffer) + " (" + std::to_string(m_Height * sizeof(float)) + " bytes)");

        // Initialize activation and delta buffers
        m_ActivationMat = std::make_shared<Matrix>(m_Height, batchSize);
        m_ActivationMat->uploadToGPU();
        LOG_DEBUG("[GPU BUFFER] Created activation buffer " + std::to_string(m_ActivationMat->buffer) + " (" + std::to_string(batchSize * m_Height * sizeof(float)) + " bytes)");

        m_PreactivationMat = std::make_shared<Matrix>(m_Height, batchSize);
        m_PreactivationMat->uploadToGPU();
        LOG_DEBUG("[GPU BUFFER] Created preactivation buffer " + std::to_string(m_PreactivationMat->buffer) + " (" + std::to_string(batchSize * m_Height * sizeof(float)) + " bytes)");

        m_DeltaMat = std::make_shared<Matrix>(m_Height, batchSize);
        m_DeltaMat->uploadToGPU();
        LOG_DEBUG("[GPU BUFFER] Created delta buffer " + std::to_string(m_DeltaMat->buffer) + " (" + std::to_string(batchSize * m_Height * sizeof(float)) + " bytes)");

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	}

	Layer::~Layer() { }

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
        // Read weights
        m_WeightMat->downloadFromGPU();
        
        // Log GPU data read for visualization
        LOG_DEBUG("[GPU DOWNLOAD] Weight data (" + std::to_string(m_WeightMat->getFlatVec().size() * sizeof(float)) +
            " bytes) downloaded from weight buffer " + std::to_string(m_WeightMat->buffer) + " for heatmap");

        // Find min/max for normalization
        auto [minIt, maxIt] = std::minmax_element(m_WeightMat->getFlatVec().begin(), m_WeightMat->getFlatVec().end());
        float minVal = *minIt, maxVal = *maxIt;

        for (int i = 0; i < m_Width; i++) {
            for (int j = 0; j < m_Height; j++) {
                // Normalize value between 0 and 1
                float normalized = (m_WeightMat->get(i, j) - minVal) / (maxVal - minVal);
                int colorIdx = static_cast<int>(normalized * (sizeof(colors) / sizeof(colors[0]) - 2));
                std::cout << colors[colorIdx] << "  " << colors[16]; // print 2 spaces with bg color, then reset
            }
            std::cout << "\n";
        }

        std::cout << "\033[0m\n";
        std::cout << std::endl;
    }
}