#include "NeuralNetwork.h"

#include <execution>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <string>
#include <tuple>

namespace NNGL {
    NeuralNetwork::NeuralNetwork(int batchSize) : m_BatchSize(batchSize), m_ADAM_Timestep(0) {
        //Neural Network run
        if (!m_ForwardPassCompute)	m_ForwardPassCompute = ShaderManager::getInstance().getShader("shaders/forward_pass.comp");

        //Backpropagation delta calc
        if (!m_OutputDeltaCompute)	m_OutputDeltaCompute = ShaderManager::getInstance().getShader("shaders/output_delta_loss.comp");
        if (!m_HiddenDeltasCompute) m_HiddenDeltasCompute = ShaderManager::getInstance().getShader("shaders/hidden_delta_loss.comp");

        //Backpropagation weights/biases update by delta
        if (!m_WeightsCompute)		m_WeightsCompute = ShaderManager::getInstance().getShader("shaders/update_weights.comp");
        if (!m_BiasesCompute)		m_BiasesCompute = ShaderManager::getInstance().getShader("shaders/update_biases.comp");
    };

	NeuralNetwork::~NeuralNetwork() {}

	void NeuralNetwork::addLayer(int width, int height,  ActivationFnType type) {
		if (!m_Layers.empty() && m_Layers.back()->getSize().y != width)
			throw std::runtime_error("Trying to chain layers with incompatible dementions: last height != new width");

		m_Layers.push_back(std::unique_ptr<NNGL::Layer>( new NNGL::Layer(width, height, m_BatchSize, type) ));

        //we changed layer structure so we need to update mat's
        m_InputBatchMat = nullptr;
        m_OutputBatchMat = nullptr;
        forwardMatOutput = nullptr;
	}

	void NeuralNetwork::forwardPass(std::shared_ptr<Matrix> &inputBatchMat) {
        inputBatchMat->uploadToGPU();
		GLuint currentInput = inputBatchMat->buffer;
		for (auto &layer : m_Layers) {

            m_ForwardPassCompute->bindBuffer(0, "InputBuffer", currentInput);
            m_ForwardPassCompute->bindBuffer(1, "WeightBuffer", layer->m_WeightBuffer);
            m_ForwardPassCompute->bindBuffer(2, "BiasBuffer", layer->m_BiasBuffer);
            m_ForwardPassCompute->bindBuffer(3, "OutputBuffer", layer->m_ActivationBuffer);
            m_ForwardPassCompute->bindBuffer(4, "PreActivationBuffer", layer->m_PreactivationBuffer);

            m_ForwardPassCompute->setUniform("input_size", (int)layer->getSize().x);
            m_ForwardPassCompute->setUniform("output_size", (int)layer->getSize().y);
            m_ForwardPassCompute->setUniform("batch_size", m_BatchSize);
            m_ForwardPassCompute->setUniform("activation_type", layer->m_ActivationFnType);

			// Safe workgroup calculation with bounds checking
			int workgroupsX = std::min((int)ceil(m_BatchSize * layer->getSize().x / 16.0f), 65535);
			int workgroupsY = std::min((int)ceil(m_BatchSize * layer->getSize().y / 16.0f), 65535);
            m_ForwardPassCompute->dispatch(workgroupsX, workgroupsY, 1);

			currentInput = layer->m_ActivationBuffer;
		}

		// Unbind buffers for this layer
		for (int j = 0; j < 5; j++) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, j, 0);
	}

	void NeuralNetwork::targetLayerLossCalc(std::shared_ptr<Matrix>& outputBatchMat) {
        outputBatchMat->uploadToGPU();

        m_OutputDeltaCompute->bindBuffer(0, "OutputBuffer", m_Layers.back()->m_ActivationBuffer);
        m_OutputDeltaCompute->bindBuffer(1, "TargetBuffer", outputBatchMat->buffer);
        m_OutputDeltaCompute->bindBuffer(2, "PreActivationBuffer", m_Layers.back()->m_PreactivationBuffer);
        m_OutputDeltaCompute->bindBuffer(3, "DeltaBuffer", m_Layers.back()->m_DeltaBuffer);

        m_OutputDeltaCompute->setUniform("output_size", (int)m_Layers.back()->getSize().y);
        m_OutputDeltaCompute->setUniform("batch_size", m_BatchSize);
        m_OutputDeltaCompute->setUniform("activation_type", m_Layers.back()->m_ActivationFnType);

        // Safe workgroup calculation with bounds checking
		int outputWorkgroups = std::min((int)ceil((m_BatchSize * m_Layers.back()->getSize().y + 31) / 32), 65535);
        m_OutputDeltaCompute->dispatch(outputWorkgroups, 1, 1);

		// Unbind output delta buffers
		for (int j = 0; j < 4; j++) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, j, 0);

	}

	void NeuralNetwork::hiddenLayersLossCalc() {
		for (int i = static_cast<int>(m_Layers.size()) - 2; i >= 0; i--) {

            m_HiddenDeltasCompute->bindBuffer(0, "PreActivationBuffer", m_Layers[i]->m_PreactivationBuffer);
            m_HiddenDeltasCompute->bindBuffer(1, "NextDeltaBuffer", m_Layers[i + 1]->m_DeltaBuffer);
            m_HiddenDeltasCompute->bindBuffer(2, "WeightBuffer", m_Layers[i + 1]->m_WeightBuffer);
            m_HiddenDeltasCompute->bindBuffer(3, "DeltaBuffer", m_Layers[i]->m_DeltaBuffer);

            m_HiddenDeltasCompute->setUniform("current_size", (int)m_Layers[i]->getSize().y);
            m_HiddenDeltasCompute->setUniform("next_size", (int)m_Layers[i + 1]->getSize().y);
            m_HiddenDeltasCompute->setUniform("batch_size", m_BatchSize);
            m_HiddenDeltasCompute->setUniform("activation_type", m_Layers[i]->m_ActivationFnType);

			// Safe workgroup calculation
			int workgroupsX = std::min((int)ceil(m_BatchSize * m_Layers[i]->getSize().x / 16.0f), 65535);
			int workgroupsY = std::min((int)ceil(m_BatchSize * m_Layers[i]->getSize().y / 16.0f), 65535);
            m_HiddenDeltasCompute->dispatch(workgroupsX, workgroupsY, 1);
		}

		// Unbind backward pass buffers
		for (int j = 0; j < 4; j++) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, j, 0);
	}

	// Update weights and biases for all layers
	void NeuralNetwork::weightsAndBiasesUpdate(std::shared_ptr<Matrix>& inputBatchMat, float learningRate) {
        inputBatchMat->uploadToGPU();
        GLuint currentInput = inputBatchMat->buffer;

        for (auto& layer : m_Layers) {

            // --- WEIGHTS UPDATE WITH ADAM ---
            {
                m_WeightsCompute->bindBuffer(0, "InputBuffer", currentInput);
                m_WeightsCompute->bindBuffer(1, "DeltaBuffer", layer->m_DeltaBuffer);
                m_WeightsCompute->bindBuffer(2, "WeightBuffer", layer->m_WeightBuffer);
                m_WeightsCompute->bindBuffer(3, "ADAM_MBuffer", layer->m_ADAM_MBuffer);
                m_WeightsCompute->bindBuffer(4, "ADAM_VBuffer", layer->m_ADAM_VBuffer);

                m_WeightsCompute->setUniform("input_size", (int)layer->getSize().x);
                m_WeightsCompute->setUniform("output_size", (int)layer->getSize().y);
                m_WeightsCompute->setUniform("batch_size", m_BatchSize);
                m_WeightsCompute->setUniform("learning_rate", learningRate);
                m_WeightsCompute->setUniform("ADAM_beta1", 0.9f);
                m_WeightsCompute->setUniform("ADAM_beta2", 0.999f);
                m_WeightsCompute->setUniform("ADAM_timestep", m_ADAM_Timestep);

                // Dispatch compute for weight updates
                int workgroupsX = std::min((int)ceil(m_BatchSize * layer->getSize().x / 16.0f), 65535);
                int workgroupsY = std::min((int)ceil(m_BatchSize * layer->getSize().y / 16.0f), 65535);
                m_WeightsCompute->dispatch(workgroupsX, workgroupsY, 1);

                // Update input for next layer
                currentInput = layer->m_ActivationBuffer;

                // Unbind weight update buffers
                for (int j = 0; j < 5; j++) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, j, 0);
            }

            // --- BIASES UPDATE (Simple Gradient Descent) ---
            {
                m_BiasesCompute->bindBuffer(0, "DeltaBuffer", layer->m_DeltaBuffer);
                m_BiasesCompute->bindBuffer(1, "BiasBuffer", layer->m_BiasBuffer);

                m_BiasesCompute->setUniform("output_size", (int)layer->getSize().y);
                m_BiasesCompute->setUniform("batch_size", m_BatchSize);
                m_BiasesCompute->setUniform("learning_rate", learningRate);

                int biasWorkgroups = std::min((int)ceil((m_BatchSize * layer->getSize().y + 31) / 32), 65535);
                m_BiasesCompute->dispatch(biasWorkgroups, 1, 1);

                // Unbind bias update buffers
                for (int j = 0; j < 2; j++) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, j, 0);
            }
        }

        // Increment timestep for Adam bias correction
        m_ADAM_Timestep++;
	}

     void NeuralNetwork::train(float learningRate) {
        if (m_Layers.size() < 3) throw std::runtime_error("3 layers minimum");

        if (!m_TrainBatchProvider) throw std::runtime_error("Please specify onTrainBatch callback");

        //if we changed layers on the fly 
        if (!m_InputBatchMat || !m_OutputBatchMat) {
            m_InputBatchMat = std::make_shared<Matrix>(m_Layers.front()->getSize().x, m_BatchSize);
            m_OutputBatchMat = std::make_shared<Matrix>(m_Layers.back()->getSize().y, m_BatchSize);
        }

        m_TrainBatchProvider(m_InputBatchMat, m_OutputBatchMat, m_BatchSize);

        forwardPass(m_InputBatchMat);

        targetLayerLossCalc(m_OutputBatchMat);

        hiddenLayersLossCalc();

        weightsAndBiasesUpdate(m_InputBatchMat, learningRate);

    }

    float NeuralNetwork::eval(int samplesToTest, bool doSoftmax) {

        //if we changed layers on the fly 
        if (!m_InputBatchMat || !m_OutputBatchMat) {
            m_InputBatchMat = std::make_shared<Matrix>(m_Layers.front()->getSize().x, m_BatchSize);
            m_OutputBatchMat = std::make_shared<Matrix>(m_Layers.back()->getSize().y, m_BatchSize);
        }

        int origBatchSize = m_BatchSize;
        m_BatchSize = 1;

        float totalRegressionError = 0.0f;
        float confidenceSum = 0.0f;  // sum of probabilities assigned to true classes
        int classificationSamples = 0;

        for (int i = 0; i < samplesToTest; ++i) {

            m_TestBatchProvider(m_InputBatchMat, m_OutputBatchMat, m_BatchSize);
            forwardPass(m_InputBatchMat);

            int outputSize = m_Layers.back()->getSize().y;
            std::vector<float> results(outputSize);
            std::vector<float> expected(outputSize);

            glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_Layers.back()->m_ActivationBuffer);
            float* mapped = (float*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
            if (mapped) {
                for (int j = 0; j < outputSize; ++j) {
                    results[j] = mapped[j];
                }
                glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
                
                // Log GPU data read for testing
                std::cout << "[GPU DOWNLOAD] Test results (" << outputSize * sizeof(float) 
                          << " bytes) downloaded from activation buffer " << m_Layers.back()->m_ActivationBuffer << std::endl;
            }

            for (int j = 0; j < outputSize; ++j)
                expected[j] = (*m_OutputBatchMat)(j, 0);

            if(doSoftmax) results = softmax(results);

            int trueClass = std::distance(expected.begin(), std::max_element(expected.begin(), expected.end()));
            int resultClass = std::distance(results.begin(), std::max_element(results.begin(), results.end()));

            // Accumulate probability assigned to the true class
            if(trueClass == resultClass)confidenceSum += 1;
            classificationSamples++;
        }

        m_BatchSize = origBatchSize;

        // Mean confidence over all classification samples, as % (0-100)
        float meanConfidence = (classificationSamples > 0) ? (confidenceSum / classificationSamples) * 100.0f : 0.0f;
        return meanConfidence;
    }

    std::shared_ptr<Matrix> NeuralNetwork::forward(std::shared_ptr<Matrix> inputMat) {

        forwardPass(inputMat);

        int outputSize = m_Layers.back()->getSize().y * m_BatchSize;
        
        // Use memory pool instead of creating new Matrix
        if(!forwardMatOutput) {
            forwardMatOutput = getMatrixFromPool(inputMat->rows, inputMat->cols);
        } else if (forwardMatOutput->rows != inputMat->rows || forwardMatOutput->cols != inputMat->cols) {
            forwardMatOutput->reset(inputMat->rows, inputMat->cols);
        }

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_Layers.back()->m_ActivationBuffer);
        float* mapped = (float*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
        if (mapped) {
            // Copy data directly to the pooled matrix
            std::memcpy(forwardMatOutput->raw(), mapped, forwardMatOutput->byteSize());
            glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
            
            // Log GPU data read
            std::cout << "[GPU DOWNLOAD] Forward pass results (" << forwardMatOutput->byteSize() 
                      << " bytes) downloaded from activation buffer " << m_Layers.back()->m_ActivationBuffer << std::endl;
        }
        else throw std::runtime_error("data failed to map");

        return forwardMatOutput;
    }

    std::shared_ptr<Matrix> NeuralNetwork::backward(std::shared_ptr<Matrix> inputMat, std::shared_ptr<Matrix> outputMat, float learningRate) {

        forward(inputMat);

        targetLayerLossCalc(outputMat);

        hiddenLayersLossCalc();

        weightsAndBiasesUpdate(inputMat, learningRate);

        // Use memory pool for input gradient matrix
        std::shared_ptr<Matrix> inputGradMat = getMatrixFromPool(inputMat->rows, inputMat->cols);
        
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_Layers.front()->m_DeltaBuffer);
        float* mapped = (float*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
        if (!mapped) throw std::runtime_error("data failed to map");

        // Copy data directly to the pooled matrix
        std::memcpy(inputGradMat->raw(), mapped, inputGradMat->byteSize());
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

        return inputGradMat; //attention block needs it
    }

    std::shared_ptr<Matrix> NeuralNetwork::backward_with_targetloss(std::shared_ptr<Matrix> inputMat, std::shared_ptr<Matrix> targetLoss, float learningRate) {

        forward(inputMat);

        setTargetLayerLoss(targetLoss);

        hiddenLayersLossCalc();

        weightsAndBiasesUpdate(inputMat, learningRate);

        // Use memory pool for input gradient matrix
        std::shared_ptr<Matrix> inputGradMat = getMatrixFromPool(inputMat->rows, inputMat->cols);
        
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_Layers.front()->m_DeltaBuffer);
        float* mapped = (float*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
        if (!mapped) throw std::runtime_error("data failed to map");

        // Copy data directly to the pooled matrix
        std::memcpy(inputGradMat->raw(), mapped, inputGradMat->byteSize());
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

        return inputGradMat; //attention block needs it
    }

    void NeuralNetwork::setTargetLayerLoss(std::shared_ptr<Matrix>& targetLoss) {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_Layers.back()->m_DeltaBuffer);
        glBufferData(GL_SHADER_STORAGE_BUFFER, targetLoss->flatVec.size() * sizeof(float), targetLoss->flatVec.data(), GL_DYNAMIC_DRAW);
        
        // Log target loss upload
        std::cout << "[GPU UPLOAD] Target loss data (" << targetLoss->flatVec.size() * sizeof(float) 
                  << " bytes) uploaded to delta buffer " << m_Layers.back()->m_DeltaBuffer << std::endl;
    }

    // Memory pool implementation
    std::shared_ptr<Matrix> NeuralNetwork::getMatrixFromPool(int rows, int cols) {
        std::lock_guard<std::mutex> lock(m_PoolMutex);
        
        if (!m_MatrixPool.empty()) {
            auto matrix = m_MatrixPool.front();
            m_MatrixPool.pop();
            
            // Reset the matrix to the required dimensions
            matrix->reset(rows, cols);
            return matrix;
        }
        
        // Create new matrix if pool is empty
        return std::make_shared<Matrix>(rows, cols);
    }

    void NeuralNetwork::returnMatrixToPool(std::shared_ptr<Matrix> matrix) {
        if (!matrix) return;
        
        std::lock_guard<std::mutex> lock(m_PoolMutex);
        
        // Keep pool size reasonable (max 10 matrices)
        if (m_MatrixPool.size() < 10) {
            m_MatrixPool.push(matrix);
        }
    }

    // Interactive Testing CLI
	void NeuralNetwork::run() {

        std::cout << "\n=== Neural Network Testing Interface ===" << std::endl;
        std::cout << "Commands:" << std::endl;
        std::cout << "  test             - Test with random batch" << std::endl;
        std::cout << "  batch <n>        - Test n random samples" << std::endl;
        std::cout << "  benchmark        - Performance benchmark" << std::endl;
        std::cout << "  quit             - Exit" << std::endl;
        std::cout << "  layer <i>        - Print layer" << std::endl;
        std::cout << "=====================================\n" << std::endl;

        int origBatchSize = m_BatchSize;
        m_BatchSize = 1;

        std::string command;
        while (true) {
            std::cout << "nn> ";
            std::cin >> command;

            if (command == "quit" || command == "exit" || command == "q") {
                break;
            }
            else if (command == "test") {
                // Generate random batch of 1
                m_TestBatchProvider(m_InputBatchMat, m_OutputBatchMat, m_BatchSize);
                forwardPass(m_InputBatchMat);

                // Read results from GPU
                int outputSize = m_Layers.back()->getSize().y;
                std::vector<float> results(outputSize);
                std::vector<float> expected(outputSize);

                glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_Layers.back()->m_ActivationBuffer);
                float* mapped = (float*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
                if (mapped) {
                    for (int i = 0; i < outputSize; i++) {
                        results[i] = mapped[i];
                    }
                    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
                    
                    // Log GPU data read for testing
                    std::cout << "[GPU DOWNLOAD] Test results (" << outputSize * sizeof(float) 
                              << " bytes) downloaded from activation buffer " << m_Layers.back()->m_ActivationBuffer << std::endl;
                }
                results = softmax(results);

                // Get expected values
                for (int i = 0; i < outputSize; i++) {
                    expected[i] = m_OutputBatchMat->flatVec[i];
                }

                // Display results
                std::cout << std::fixed << std::setprecision(6);

                if (outputSize == 1) {
                    // Single output (regression)
                    float error = std::abs(results[0] - expected[0]);
                    float error_percent = (expected[0] != 0) ? (error / std::abs(expected[0])) * 100.0f : error * 100.0f;

                    std::cout << "Expected: " << expected[0] << std::endl;
                    std::cout << "Got:      " << results[0] << std::endl;
                    std::cout << "Error:    " << error << " (" << error_percent << "%)" << std::endl;

                }
                else {
                    // Multiple outputs (classification or multi-output regression)
                    std::cout << "Expected: [";
                    for (int i = 0; i < outputSize; i++) {
                        std::cout << expected[i] << (i < outputSize - 1 ? ", " : "");
                    }
                    std::cout << "]" << std::endl;

                    std::cout << "Got:      [";
                    for (int i = 0; i < outputSize; i++) {
                        std::cout << results[i] << (i < outputSize - 1 ? ", " : "");
                    }
                    std::cout << "]" << std::endl;

                    // For classification, show predicted vs expected class
                    if (outputSize > 2) {  // Likely classification
                        int predictedClass = std::max_element(results.begin(), results.end()) - results.begin();
                        int expectedClass = std::max_element(expected.begin(), expected.end()) - expected.begin();

                        std::cout << "Predicted class: " << predictedClass << " (confidence: " << results[predictedClass] << ")" << std::endl;
                        std::cout << "Expected class:  " << expectedClass << " - " << (predictedClass == expectedClass ? "CORRECT" : "WRONG") << std::endl;
                    }

                    // Calculate mean squared error
                    float mse = 0.0f;
                    for (int i = 0; i < outputSize; i++) {
                        float diff = results[i] - expected[i];
                        mse += diff * diff;
                    }
                    mse /= outputSize;
                    std::cout << "MSE: " << mse << std::endl;
                }

                std::cout << std::defaultfloat << std::endl;
            }
            else if (command == "batch") {
                int n;
                if (std::cin >> n && n > 0 && n <= 1000) {
                    std::cout << "Testing " << n << " random samples..." << std::endl;

                    float total_error = 0.0f;
                    float max_error = 0.0f;
                    float min_error = std::numeric_limits<float>::max();

                    int outputSize = m_Layers.back()->getSize().y;
                    int correct_predictions = 0;

                    for (int i = 0; i < n; i++) {
                        m_TestBatchProvider(m_InputBatchMat, m_OutputBatchMat, m_BatchSize);
                        forwardPass(m_InputBatchMat);

                        std::vector<float> results(outputSize);
                        std::vector<float> expected(outputSize);

                        glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_Layers.back()->m_ActivationBuffer);
                        float* mapped = (float*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
                        if (mapped) {
                            for (int j = 0; j < outputSize; j++) {
                                results[j] = mapped[j];
                            }
                            glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
                        }
                        results = softmax(results);

                        for (int j = 0; j < outputSize; j++) {
                            expected[j] = m_OutputBatchMat->flatVec[j];
                        }

                        if (outputSize == 1) {
                            // Single output - calculate absolute error
                            float error = std::abs(results[0] - expected[0]);
                            total_error += error;
                            max_error = std::max(max_error, error);
                            min_error = std::min(min_error, error);
                        }
                        else {
                            // Multiple outputs - check if prediction is correct
                            int predictedClass = std::max_element(results.begin(), results.end()) - results.begin();
                            int expectedClass = std::max_element(expected.begin(), expected.end()) - expected.begin();

                            if (predictedClass == expectedClass) {
                                correct_predictions++;
                            }

                            // Calculate MSE for this sample
                            float mse = 0.0f;
                            for (int j = 0; j < outputSize; j++) {
                                float diff = results[j] - expected[j];
                                mse += diff * diff;
                            }
                            mse /= outputSize;
                            total_error += mse;
                        }
                    }

                    std::cout << std::fixed << std::setprecision(6);
                    if (outputSize == 1) {
                        std::cout << "Average error: " << (total_error / n) << std::endl;
                        std::cout << "Max error: " << max_error << std::endl;
                        std::cout << "Min error: " << min_error << std::endl;
                    }
                    else {
                        std::cout << "Accuracy: " << (correct_predictions * 100.0f / n) << "%" << std::endl;
                        std::cout << "Average MSE: " << (total_error / n) << std::endl;
                    }
                    std::cout << std::defaultfloat << std::endl;
                }
                else {
                    std::cout << "Invalid number. Please enter a number between 1 and 1000." << std::endl;
                }
            }
            else if (command == "benchmark") {
                std::cout << "Running performance benchmark..." << std::endl;

                const int numIterations = 1000;
                auto start = std::chrono::high_resolution_clock::now();

                for (int i = 0; i < numIterations; i++) {
                    m_TestBatchProvider(m_InputBatchMat, m_OutputBatchMat, m_BatchSize);
                    forwardPass(m_InputBatchMat);
                }

                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

                std::cout << "Benchmark completed:" << std::endl;
                std::cout << "  Iterations: " << numIterations << std::endl;
                std::cout << "  Total time: " << duration.count() << " microseconds" << std::endl;
                std::cout << "  Average time per iteration: " << (duration.count() / numIterations) << " microseconds" << std::endl;
                std::cout << "  Throughput: " << (numIterations * 1000000.0 / duration.count()) << " iterations/second" << std::endl;
            }
            else if (command == "layer") {
                int layerIndex;
                if (std::cin >> layerIndex && layerIndex >= 0 && layerIndex < m_Layers.size()) {
                    std::cout << "Layer " << layerIndex << ":" << std::endl;
                    std::cout << "  Size: " << m_Layers[layerIndex]->getSize().x << "x" << m_Layers[layerIndex]->getSize().y << std::endl;
                    std::cout << "  Activation: " << m_Layers[layerIndex]->m_ActivationFnType << std::endl;
                }
                else {
                    std::cout << "Invalid layer index. Please enter a number between 0 and " << (m_Layers.size() - 1) << "." << std::endl;
                }
            }
            else {
                std::cout << "Unknown command. Type 'help' for available commands." << std::endl;
            }
        }

        m_BatchSize = origBatchSize;
    }
}