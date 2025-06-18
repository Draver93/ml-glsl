#include "NeuralNetwork.h"

#include <execution>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <string>
#include <tuple>

namespace NNGL {
	NeuralNetwork::~NeuralNetwork() {
        if(m_InputBuffer) glDeleteBuffers(1, &m_InputBuffer);
        if(m_TargetBuffer) glDeleteBuffers(1, &m_TargetBuffer);
	}

	void NeuralNetwork::addLayer(int width, int height,  ActivationFnType type) {
		if (!m_Layers.empty() && m_Layers.back()->getSize().y != width)
			throw std::runtime_error("Trying to chain layers with incompatible dementions: last height != new width");

		m_Layers.push_back(std::unique_ptr<NNGL::Layer>( new NNGL::Layer(width, height, m_BatchSize, type) ));
	}

	void NeuralNetwork::bindTrainingData() {
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_InputBuffer);
		GLint inputBufferSize;
		glGetBufferParameteriv(GL_SHADER_STORAGE_BUFFER, GL_BUFFER_SIZE, &inputBufferSize);
		if (m_InputVector.size() * sizeof(float) > inputBufferSize)
			throw std::runtime_error("Input buffer overflow! Trying to upload " + std::to_string(m_InputVector.size() * sizeof(float)) + " bytes to buffer of size " + std::to_string(inputBufferSize) + " bytes");
		glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, m_InputVector.size() * sizeof(float), m_InputVector.data());


		glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_TargetBuffer);
		GLint targetBufferSize;
		glGetBufferParameteriv(GL_SHADER_STORAGE_BUFFER, GL_BUFFER_SIZE, &targetBufferSize);
		if (m_TargetVector.size() * sizeof(float) > targetBufferSize)
			throw std::runtime_error("Target buffer overflow! Trying to upload " + std::to_string(m_TargetVector.size() * sizeof(float)) + " bytes to buffer of size " + std::to_string(targetBufferSize) + " bytes");
		glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, m_TargetVector.size() * sizeof(float), m_TargetVector.data());


		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	}

	void NeuralNetwork::forwardPass() {
		GLuint current_input = m_InputBuffer;
		for (auto &layer : m_Layers) {

            m_ForwardPassCompute->bindBuffer(0, "InputBuffer", current_input);
            m_ForwardPassCompute->bindBuffer(1, "WeightBuffer", layer->m_WeightBuffer);
            m_ForwardPassCompute->bindBuffer(2, "BiasBuffer", layer->m_BiasBuffer);
            m_ForwardPassCompute->bindBuffer(3, "OutputBuffer", layer->m_ActivationBuffer);
            m_ForwardPassCompute->bindBuffer(4, "PreActivationBuffer", layer->m_PreactivationBuffer);

            m_ForwardPassCompute->setUniform("input_size", (int)layer->getSize().x);
            m_ForwardPassCompute->setUniform("output_size", (int)layer->getSize().y);
            m_ForwardPassCompute->setUniform("batch_size", m_BatchSize);
            m_ForwardPassCompute->setUniform("activation_type", layer->m_ActivationFnType);

			// Safe workgroup calculation with bounds checking
			int workgroups_x = std::min((int)ceil(m_BatchSize * layer->getSize().x / 16.0f), 65535);
			int workgroups_y = std::min((int)ceil(m_BatchSize * layer->getSize().y / 16.0f), 65535);
            m_ForwardPassCompute->dispatch(workgroups_x, workgroups_y, 1);

			current_input = layer->m_ActivationBuffer;
		}

		// Unbind buffers for this layer
		for (int j = 0; j < 5; j++) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, j, 0);
	}

	void NeuralNetwork::targetLayerLossCalc() {

        m_OutputDeltaCompute->bindBuffer(0, "OutputBuffer", m_Layers.back()->m_ActivationBuffer);
        m_OutputDeltaCompute->bindBuffer(1, "TargetBuffer", m_TargetBuffer);
        m_OutputDeltaCompute->bindBuffer(2, "PreActivationBuffer", m_Layers.back()->m_PreactivationBuffer);
        m_OutputDeltaCompute->bindBuffer(3, "DeltaBuffer", m_Layers.back()->m_DeltaBuffer);

        m_OutputDeltaCompute->setUniform("output_size", (int)m_Layers.back()->getSize().y);
        m_OutputDeltaCompute->setUniform("batch_size", m_BatchSize);
        m_OutputDeltaCompute->setUniform("activation_type", m_Layers.back()->m_ActivationFnType);

        // Safe workgroup calculation with bounds checking
		int output_workgroups = std::min((int)ceil((m_BatchSize * m_Layers.back()->getSize().y + 31) / 32), 65535);
        m_OutputDeltaCompute->dispatch(output_workgroups, 1, 1);

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
			int workgroups_x = std::min((int)ceil(m_BatchSize * m_Layers[i]->getSize().x / 16.0f), 65535);
			int workgroups_y = std::min((int)ceil(m_BatchSize * m_Layers[i]->getSize().y / 16.0f), 65535);
            m_HiddenDeltasCompute->dispatch(workgroups_x, workgroups_y, 1);
		}

		// Unbind backward pass buffers
		for (int j = 0; j < 4; j++) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, j, 0);
	}

	// Update weights and biases for all layers
	void NeuralNetwork::weightsAndBiasesUpdate(float learningRate) {
        GLuint current_input = m_InputBuffer;

        for (auto& layer : m_Layers) {

            // --- WEIGHTS UPDATE WITH ADAM ---
            {
                m_WeightsCompute->bindBuffer(0, "InputBuffer", current_input);
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
                int workgroups_x = std::min((int)ceil(m_BatchSize * layer->getSize().x / 16.0f), 65535);
                int workgroups_y = std::min((int)ceil(m_BatchSize * layer->getSize().y / 16.0f), 65535);
                m_WeightsCompute->dispatch(workgroups_x, workgroups_y, 1);

                // Update input for next layer
                current_input = layer->m_ActivationBuffer;

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

                int bias_workgroups = std::min((int)ceil((m_BatchSize * layer->getSize().y + 31) / 32), 65535);
                m_BiasesCompute->dispatch(bias_workgroups, 1, 1);

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

        if (m_InputBuffer == 0 || m_TargetBuffer == 0) init();

        m_TrainBatchProvider(m_InputVector, m_TargetVector, m_BatchSize);

        bindTrainingData();

        forwardPass();

        targetLayerLossCalc();

        hiddenLayersLossCalc();

        weightsAndBiasesUpdate(learningRate);

    }

	void NeuralNetwork::init()
	{
		if (m_Layers.size() < 3) throw std::runtime_error("3 layers minimum");

		if (m_InputBuffer != 0) glDeleteBuffers(1, &m_InputBuffer);
		if (m_TargetBuffer != 0) glDeleteBuffers(1, &m_TargetBuffer);

        m_InputVector.clear();
        m_InputVector.resize(m_Layers.front()->getSize().x * m_BatchSize);

        m_TargetVector.clear();
        m_TargetVector.resize(m_Layers.back()->getSize().y * m_BatchSize);

		glGenBuffers(1, &m_InputBuffer);
		glGenBuffers(1, &m_TargetBuffer);

		glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_InputBuffer);
		glBufferData(GL_SHADER_STORAGE_BUFFER, m_BatchSize * m_Layers.front()->getSize().x * sizeof(float), nullptr, GL_DYNAMIC_DRAW);

		glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_TargetBuffer);
		glBufferData(GL_SHADER_STORAGE_BUFFER, m_Layers.back()->getSize().y * m_BatchSize * sizeof(float), nullptr, GL_DYNAMIC_DRAW);

        //Neural Network run
		if (!m_ForwardPassCompute)	m_ForwardPassCompute = std::make_unique<Shader>("shaders/forward_pass.comp");

        //Backpropagation delta calc
		if (!m_OutputDeltaCompute)	m_OutputDeltaCompute = std::make_unique<Shader>("shaders/output_delta_loss.comp");
		if (!m_HiddenDeltasCompute) m_HiddenDeltasCompute = std::make_unique<Shader>("shaders/hidden_delta_loss.comp");

        //Backpropagation weights/biases update by delta
		if (!m_WeightsCompute)		m_WeightsCompute = std::make_unique<Shader>("shaders/update_weights.comp");
		if (!m_BiasesCompute)		m_BiasesCompute = std::make_unique<Shader>("shaders/update_biases.comp");
	}
    
float NeuralNetwork::eval(int samplesToTest, bool do_softmax) {

    if (m_InputBuffer == 0 || m_TargetBuffer == 0) init();

    int origBatchSize = m_BatchSize;
    m_BatchSize = 1;

    float totalRegressionError = 0.0f;
    float confidenceSum = 0.0f;  // sum of probabilities assigned to true classes
    int classificationSamples = 0;

    if (m_InputBuffer == 0 || m_TargetBuffer == 0) init();

    for (int i = 0; i < samplesToTest; ++i) {
        m_TestBatchProvider(m_InputVector, m_TargetVector, m_BatchSize);
        bindTrainingData();
        forwardPass();

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
        }

        for (int j = 0; j < outputSize; ++j)
            expected[j] = m_TargetVector[j];

        if(do_softmax) results = softmax(results);

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
                m_TestBatchProvider(m_InputVector, m_TargetVector, m_BatchSize);
                bindTrainingData();
                forwardPass();

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
                }
                results = softmax(results);

                // Get expected values
                for (int i = 0; i < outputSize; i++) {
                    expected[i] = m_TargetVector[i];
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
                        m_TestBatchProvider(m_InputVector, m_TargetVector, m_BatchSize);
                        bindTrainingData();
                        forwardPass();

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
                            expected[j] = m_TargetVector[j];
                        }

                        if (outputSize == 1) {
                            // Single output - calculate absolute error
                            float error = std::abs(results[0] - expected[0]);
                            total_error += error;
                            max_error = std::max(max_error, error);
                            min_error = std::min(min_error, error);
                        }
                        else {
                            // Multiple outputs - calculate MSE and check classification accuracy
                            float mse = 0.0f;
                            for (int j = 0; j < outputSize; j++) {
                                float diff = results[j] - expected[j];
                                mse += diff * diff;
                            }
                            mse /= outputSize;
                            total_error += mse;
                            max_error = std::max(max_error, mse);
                            min_error = std::min(min_error, mse);

                            // Check classification accuracy (if likely classification)
                            if (outputSize > 2) {
                                int predictedClass = std::max_element(results.begin(), results.end()) - results.begin();
                                int expectedClass = std::max_element(expected.begin(), expected.end()) - expected.begin();
                                if (predictedClass == expectedClass) {
                                    correct_predictions++;
                                }
                            }
                        }
                    }

                    std::cout << std::fixed << std::setprecision(6);
                    std::cout << "Results for " << n << " samples:" << std::endl;

                    if (outputSize == 1) {
                        std::cout << "Average error: " << (total_error / n) << std::endl;
                        std::cout << "Min error:     " << min_error << std::endl;
                        std::cout << "Max error:     " << max_error << std::endl;
                    }
                    else {
                        std::cout << "Average MSE:   " << (total_error / n) << std::endl;
                        std::cout << "Min MSE:       " << min_error << std::endl;
                        std::cout << "Max MSE:       " << max_error << std::endl;

                        if (outputSize > 2) {
                            float accuracy = (float)correct_predictions / n * 100.0f;
                            std::cout << "Accuracy:      " << accuracy << "% (" << correct_predictions << "/" << n << " correct)" << std::endl;
                        }
                    }

                    std::cout << std::defaultfloat << std::endl;
                }
                else {
                    std::cout << "Usage: batch <n> (where n is 1-1000)" << std::endl;
                    std::cin.clear();
                    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                }
            }
            else if (command == "benchmark") {
                std::cout << "Running performance benchmark..." << std::endl;

                const int benchmark_samples = 10000;
                auto start_time = std::chrono::high_resolution_clock::now();

                for (int i = 0; i < benchmark_samples; i++) {
                    m_TestBatchProvider(m_InputVector, m_TargetVector, m_BatchSize);
                    bindTrainingData();
                    forwardPass();
                }

                glFinish(); // Wait for all GPU operations to complete
                auto end_time = std::chrono::high_resolution_clock::now();

                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
                double seconds = duration.count() / 1000000.0;
                double samples_per_second = benchmark_samples / seconds;

                std::cout << std::fixed << std::setprecision(2);
                std::cout << "Benchmark results:" << std::endl;
                std::cout << "Processed " << benchmark_samples << " samples in " << seconds << " seconds" << std::endl;
                std::cout << "Performance: " << samples_per_second << " inferences/second" << std::endl;
                std::cout << "Average time per inference: " << (seconds * 1000000 / benchmark_samples) << " microseconds" << std::endl;
                std::cout << std::defaultfloat << std::endl;
            }
            else if (command == "help" || command == "h") {
                std::cout << "Available commands:" << std::endl;
                std::cout << "  test             - Test with random sample" << std::endl;
                std::cout << "  batch <n>        - Test n random samples (1-1000)" << std::endl;
                std::cout << "  benchmark        - Performance benchmark (10k samples)" << std::endl;
                std::cout << "  layer <i>        - Show layer info" << std::endl;
                std::cout << "  help             - Show this help" << std::endl;
                std::cout << "  quit             - Exit" << std::endl;
            }
            else if (command == "layer") {
                int layer_idx;
                if (!(std::cin >> layer_idx) || layer_idx < 0 || layer_idx >= m_Layers.size()) {
                    std::cout << "Usage: layer <0-" << (m_Layers.size() - 1) << ">" << std::endl;
                    continue;
                }
                m_Layers[layer_idx]->displayLayer("Layer " + std::to_string(layer_idx));
            }
            else {
                std::cout << "Unknown command: " << command << std::endl;
                std::cout << "Type 'help' for available commands." << std::endl;
            }

            std::cin.clear();
        }

        m_BatchSize = origBatchSize;
	}
}