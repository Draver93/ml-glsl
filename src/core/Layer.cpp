#include "Layer.h"
#include "Logger.h"

#include <execution>
#include <string>
#include <iostream>

namespace MLGL {
	Layer::Layer(int width, int height, int batchSize, ActivationFnType type) 
        : m_Width(width), m_Height(height), m_BatchSize(batchSize), m_ActivationFnType(type) {

        LOG_DEBUG("[LAYER INIT] Creating layer " + std::to_string(m_Width) + "x" + std::to_string(m_Height) + " with batch size " + std::to_string(m_ActivationFnType));

        // Initialize biases
        m_WeightMat = std::make_shared<Matrix>(m_Width, m_Height);
        for (int i = 0; i < m_WeightMat->rows; i++)
            for (int j = 0; j < m_WeightMat->cols; j++)
                m_WeightMat->set(i, j, MLGL::activationFunctions[m_ActivationFnType].weight_initializer(m_Width, m_Height));
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
        for (int i = 0; i < m_BiasMat->rows; i++) m_BiasMat->set(i, 0, MLGL::activationFunctions[m_ActivationFnType].weight_initializer(m_Width, m_Height));
        m_BiasMat->uploadToGPU();
        LOG_DEBUG("[GPU BUFFER] Created bias buffer " + std::to_string(m_BiasMat->buffer) + " (" + std::to_string(m_Height * sizeof(float)) + " bytes)");

        // Initialize activation and delta buffers
        m_ActivationMat = std::make_shared<Matrix>(m_Height, m_BatchSize);
        m_ActivationMat->uploadToGPU();
        LOG_DEBUG("[GPU BUFFER] Created activation buffer " + std::to_string(m_ActivationMat->buffer) + " (" + std::to_string(m_BatchSize * m_Height * sizeof(float)) + " bytes)");

        m_PreactivationMat = std::make_shared<Matrix>(m_Height, m_BatchSize);
        m_PreactivationMat->uploadToGPU();
        LOG_DEBUG("[GPU BUFFER] Created preactivation buffer " + std::to_string(m_PreactivationMat->buffer) + " (" + std::to_string(m_BatchSize * m_Height * sizeof(float)) + " bytes)");

        m_DeltaMat = std::make_shared<Matrix>(m_Height, m_BatchSize);
        m_DeltaMat->uploadToGPU();
        LOG_DEBUG("[GPU BUFFER] Created delta buffer " + std::to_string(m_DeltaMat->buffer) + " (" + std::to_string(m_BatchSize * m_Height * sizeof(float)) + " bytes)");

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	}

    Layer::Layer(const char* data) {
        if (!data) {
            throw std::invalid_argument("Data pointer cannot be null");
        }

        const char* ptr = data;

        // Read dimensions and activation type
        std::memcpy(&m_Width, ptr, sizeof(int));
        ptr += sizeof(int);

        std::memcpy(&m_Height, ptr, sizeof(int));
        ptr += sizeof(int);

        std::memcpy(&m_BatchSize, ptr, sizeof(int));
        ptr += sizeof(int);

        std::memcpy(&m_ActivationFnType, ptr, sizeof(ActivationFnType));
        ptr += sizeof(ActivationFnType);

        LOG_DEBUG("[LAYER LOAD] Loading layer " + std::to_string(m_Width) + "x" + std::to_string(m_Height) + " with activation type " + std::to_string(m_ActivationFnType));

        // Read weight matrix
        int weight_data_size;
        std::memcpy(&weight_data_size, ptr, sizeof(int));
        ptr += sizeof(int);

        if (weight_data_size != m_Width * m_Height * sizeof(float)) {
            throw std::runtime_error("Weight data size mismatch during layer loading");
        }

        m_WeightMat = std::make_shared<Matrix>(m_Width, m_Height, reinterpret_cast<const float*>(ptr));
        ptr += weight_data_size;
        m_WeightMat->uploadToGPU();
        LOG_DEBUG("[GPU BUFFER] Loaded weight buffer " + std::to_string(m_WeightMat->buffer) + " (" + std::to_string(weight_data_size) + " bytes)");

        // Read bias matrix
        int bias_data_size;
        std::memcpy(&bias_data_size, ptr, sizeof(int));
        ptr += sizeof(int);

        if (bias_data_size != m_Height * sizeof(float)) throw std::runtime_error("Bias data size mismatch during layer loading");
 
        m_BiasMat = std::make_shared<Matrix>(m_Height, 1, reinterpret_cast<const float*>(ptr));
        ptr += bias_data_size;
        m_BiasMat->uploadToGPU();
        LOG_DEBUG("[GPU BUFFER] Loaded bias buffer " + std::to_string(m_BiasMat->buffer) + " (" + std::to_string(bias_data_size) + " bytes)");

        // Initialize ADAM matrices (these are not saved, always start fresh)
        m_ADAM_M_Mat = std::make_shared<Matrix>(m_Width, m_Height, 0.0f);
        m_ADAM_M_Mat->uploadToGPU();
        LOG_DEBUG("[GPU BUFFER] Created fresh ADAM M buffer " + std::to_string(m_ADAM_M_Mat->buffer));

        m_ADAM_V_Mat = std::make_shared<Matrix>(m_Width, m_Height, 0.0f);
        m_ADAM_V_Mat->uploadToGPU();
        LOG_DEBUG("[GPU BUFFER] Created fresh ADAM V buffer " + std::to_string(m_ADAM_V_Mat->buffer));

        // Initialize activation and delta buffers
        m_ActivationMat = std::make_shared<Matrix>(m_Height, m_BatchSize);
        m_ActivationMat->uploadToGPU();
        LOG_DEBUG("[GPU BUFFER] Created activation buffer " + std::to_string(m_ActivationMat->buffer) + " (" + std::to_string(m_BatchSize * m_Height * sizeof(float)) + " bytes)");

        m_PreactivationMat = std::make_shared<Matrix>(m_Height, m_BatchSize);
        m_PreactivationMat->uploadToGPU();
        LOG_DEBUG("[GPU BUFFER] Created preactivation buffer " + std::to_string(m_PreactivationMat->buffer) + " (" + std::to_string(m_BatchSize * m_Height * sizeof(float)) + " bytes)");

        m_DeltaMat = std::make_shared<Matrix>(m_Height, m_BatchSize);
        m_DeltaMat->uploadToGPU();
        LOG_DEBUG("[GPU BUFFER] Created delta buffer " + std::to_string(m_DeltaMat->buffer) + " (" + std::to_string(m_BatchSize * m_Height * sizeof(float)) + " bytes)");

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

        LOG_DEBUG("[LAYER LOAD] Layer loaded successfully");
    }

    const char* Layer::save() {
        // Download data from GPU first
        m_WeightMat->downloadFromGPU();
        m_BiasMat->downloadFromGPU();

        LOG_DEBUG("[LAYER SAVE] Saving layer " + std::to_string(m_Width) + "x" + std::to_string(m_Height));

        // Calculate total buffer size needed
        size_t header_size = 4 * sizeof(int) + sizeof(ActivationFnType);
        size_t weight_size = m_WeightMat->byteSize();
        size_t bias_size = m_BiasMat->byteSize();
        size_t total_size = header_size + 2 * sizeof(int) + weight_size + bias_size;

        // Allocate buffer (caller is responsible for freeing this memory)
        char* buffer = new char[total_size];
        char* ptr = buffer;

        // Write header: dimensions and activation type
        std::memcpy(ptr, &m_Width, sizeof(int));
        ptr += sizeof(int);

        std::memcpy(ptr, &m_Height, sizeof(int));
        ptr += sizeof(int);

        std::memcpy(ptr, &m_BatchSize, sizeof(int));
        ptr += sizeof(int);

        std::memcpy(ptr, &m_ActivationFnType, sizeof(ActivationFnType));
        ptr += sizeof(ActivationFnType);

        // Write weight matrix
        int weight_data_size = static_cast<int>(weight_size);
        std::memcpy(ptr, &weight_data_size, sizeof(int));
        ptr += sizeof(int);

        std::memcpy(ptr, m_WeightMat->raw(), weight_size);
        ptr += weight_size;

        // Write bias matrix
        int bias_data_size = static_cast<int>(bias_size);
        std::memcpy(ptr, &bias_data_size, sizeof(int));
        ptr += sizeof(int);

        std::memcpy(ptr, m_BiasMat->raw(), bias_size);
        ptr += bias_size;

        LOG_DEBUG("[LAYER SAVE] Layer saved to binary buffer (" + std::to_string(total_size) + " bytes total)");
        LOG_DEBUG("[LAYER SAVE] Weight data: " + std::to_string(weight_size) + " bytes");
        LOG_DEBUG("[LAYER SAVE] Bias data: " + std::to_string(bias_size) + " bytes");

        return buffer;
    }

    // Helper method to get the size of the saved data (useful for file I/O)
    size_t Layer::getSaveSize() const {
        size_t header_size = 4 * sizeof(int) + sizeof(ActivationFnType);
        size_t weight_size = m_WeightMat->byteSize();
        size_t bias_size = m_BiasMat->byteSize();
        return header_size + 2 * sizeof(int) + weight_size + bias_size;
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