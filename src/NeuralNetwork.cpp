#include "NeuralNetwork.h"

#include <execution>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <string>
#include <tuple>

namespace NNGL {
	NeuralNetwork::NeuralNetwork() {
	}

	NeuralNetwork::~NeuralNetwork() {
        if(m_InputBuffer) glDeleteBuffers(1, &m_InputBuffer);
        if(m_TargetBuffer) glDeleteBuffers(1, &m_TargetBuffer);
	}

	void NeuralNetwork::addLayer(std::unique_ptr<Layer> layer) {
		if (!m_Layers.empty() && m_Layers.back()->getSize().y != layer->getSize().x )
			throw std::runtime_error("Trying to chain layers with incompatible dementions: last height != new width");

		m_Layers.push_back(std::move(layer));
	}

	void NeuralNetwork::bindTrainingData(const std::vector<float>& inputBatch, const std::vector<float>& targetBatch) {
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_InputBuffer);
		GLint inputBufferSize;
		glGetBufferParameteriv(GL_SHADER_STORAGE_BUFFER, GL_BUFFER_SIZE, &inputBufferSize);
		if (inputBatch.size() * sizeof(float) > inputBufferSize)
			throw std::runtime_error("Input buffer overflow! Trying to upload " + std::to_string(inputBatch.size() * sizeof(float)) + " bytes to buffer of size " + std::to_string(inputBufferSize) + " bytes");
		glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, inputBatch.size() * sizeof(float), inputBatch.data());


		glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_TargetBuffer);
		GLint targetBufferSize;
		glGetBufferParameteriv(GL_SHADER_STORAGE_BUFFER, GL_BUFFER_SIZE, &targetBufferSize);
		if (targetBatch.size() * sizeof(float) > targetBufferSize)
			throw std::runtime_error("Target buffer overflow! Trying to upload " + std::to_string(targetBatch.size() * sizeof(float)) + " bytes to buffer of size " + std::to_string(targetBufferSize) + " bytes");
		glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, targetBatch.size() * sizeof(float), targetBatch.data());


		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	}

	void NeuralNetwork::forwardPass() {
		GLuint current_input = m_InputBuffer;
		for (auto &layer : m_Layers) {
			glUseProgram(m_ForwardPassCompute->get());

			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, current_input);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, layer->m_WeightBuffer);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, layer->m_BiasBuffer);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, layer->m_ActivationBuffer);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, layer->m_PreactivationBuffer);

			glUniform1i(glGetUniformLocation(m_ForwardPassCompute->get(), "input_size"), layer->getSize().x);
			glUniform1i(glGetUniformLocation(m_ForwardPassCompute->get(), "output_size"), layer->getSize().y);
			glUniform1i(glGetUniformLocation(m_ForwardPassCompute->get(), "batch_size"), layer->getBatchSize());
			glUniform1i(glGetUniformLocation(m_ForwardPassCompute->get(), "activation_type"), layer->m_ActivationFnType);

			// Safe workgroup calculation with bounds checking
			int workgroups_x = std::min((int)ceil(layer->getBatchSize() * layer->getSize().x / 16.0f), 65535);
			int workgroups_y = std::min((int)ceil(layer->getBatchSize() * layer->getSize().y / 16.0f), 65535);
			glDispatchCompute(workgroups_x, workgroups_y, 1);
			glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

			current_input = layer->m_ActivationBuffer;
		}

		// Unbind buffers for this layer
		for (int j = 0; j < 5; j++) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, j, 0);
	}

	void NeuralNetwork::targetLayerLossCalc() {
		glUseProgram(m_OutputDeltaCompute->get());

		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_Layers.back()->m_ActivationBuffer);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_TargetBuffer);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_Layers.back()->m_PreactivationBuffer);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, m_Layers.back()->m_DeltaBuffer);

		glUniform1i(glGetUniformLocation(m_OutputDeltaCompute->get(), "output_size"), m_Layers.back()->getSize().y);
		glUniform1i(glGetUniformLocation(m_OutputDeltaCompute->get(), "batch_size"), m_Layers.back()->getBatchSize());
		glUniform1i(glGetUniformLocation(m_OutputDeltaCompute->get(), "activation_type"), m_Layers.back()->m_ActivationFnType);

		int output_workgroups = std::min((int)ceil((m_Layers.back()->getBatchSize() * m_Layers.back()->getSize().y + 31) / 32), 65535);
		glDispatchCompute(output_workgroups, 1, 1);
		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

		// Unbind output delta buffers
		for (int j = 0; j < 4; j++) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, j, 0);

	}

	void NeuralNetwork::hiddenLayersLossCalc() {
		for (int i = static_cast<int>(m_Layers.size()) - 2; i >= 0; i--) {

			glUseProgram(m_HiddenDeltasCompute->get());

			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_Layers[i]->m_PreactivationBuffer);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_Layers[i + 1]->m_DeltaBuffer);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_Layers[i + 1]->m_WeightBuffer);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, m_Layers[i]->m_DeltaBuffer);

			glUniform1i(glGetUniformLocation(m_HiddenDeltasCompute->get(), "current_size"), m_Layers[i]->getSize().y);
			glUniform1i(glGetUniformLocation(m_HiddenDeltasCompute->get(), "next_size"), m_Layers[i + 1]->getSize().y);
			glUniform1i(glGetUniformLocation(m_HiddenDeltasCompute->get(), "batch_size"), m_Layers[i]->getBatchSize());
			glUniform1i(glGetUniformLocation(m_HiddenDeltasCompute->get(), "activation_type"), m_Layers[i]->m_ActivationFnType);

			// Safe workgroup calculation
			int workgroups_x = std::min((int)ceil(m_Layers[i]->getBatchSize() * m_Layers[i]->getSize().x / 16.0f), 65535);
			int workgroups_y = std::min((int)ceil(m_Layers[i]->getBatchSize() * m_Layers[i]->getSize().y / 16.0f), 65535);
			glDispatchCompute(workgroups_x, workgroups_y, 1);
			glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
		}

		// Unbind backward pass buffers
		for (int j = 0; j < 4; j++) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, j, 0);
	}

	// Update weights and biases for all layers
	void NeuralNetwork::weightsAndBiasesUpdate(float learningRate) {
		GLuint current_input = m_InputBuffer;
		for (auto &layer : m_Layers) {
			glUseProgram(m_WeightsCompute->get());

			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, current_input);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, layer->m_DeltaBuffer);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, layer->m_WeightBuffer);

			glUniform1i(glGetUniformLocation(m_WeightsCompute->get(), "input_size"), layer->getSize().x);
			glUniform1i(glGetUniformLocation(m_WeightsCompute->get(), "output_size"), layer->getSize().y);
			glUniform1i(glGetUniformLocation(m_WeightsCompute->get(), "batch_size"), layer->getBatchSize());
			glUniform1f(glGetUniformLocation(m_WeightsCompute->get(), "learning_rate"), learningRate);

			int workgroups_x = std::min((int)ceil(layer->getBatchSize() * layer->getSize().x / 16.0f), 65535);
			int workgroups_y = std::min((int)ceil(layer->getBatchSize() * layer->getSize().y / 16.0f), 65535);
			glDispatchCompute(workgroups_x, workgroups_y, 1);
			glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

			// Unbind weight update buffers
			for (int j = 0; j < 3; j++) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, j, 0);

			// Update biases
			glUseProgram(m_BiasesCompute->get());

			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, layer->m_DeltaBuffer);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, layer->m_BiasBuffer);

			glUniform1i(glGetUniformLocation(m_BiasesCompute->get(), "output_size"), layer->getSize().y);
			glUniform1i(glGetUniformLocation(m_BiasesCompute->get(), "batch_size"), layer->getBatchSize());
			glUniform1f(glGetUniformLocation(m_BiasesCompute->get(), "learning_rate"), learningRate);

			int bias_workgroups = std::min((int)ceil((layer->getBatchSize() * layer->getSize().y + 31) / 32), 65535);
			glDispatchCompute(bias_workgroups, 1, 1);
			glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

			// Unbind bias update buffers
			for (int j = 0; j < 2; j++) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, j, 0);

			current_input = layer->m_ActivationBuffer;
		}
	}

	void NeuralNetwork::train(const std::vector<float>& inputBatch, const std::vector<float>& targetBatch, float learningRate) {
		if (m_Layers.size() < 3) throw std::runtime_error("3 layers minimum");

		if (inputBatch.size() != m_Layers.front()->getSize().x * m_Layers.front()->getBatchSize())
			throw std::runtime_error("Wrong input_batch size! Should be: " + std::to_string(m_Layers.front()->getSize().x * m_Layers.front()->getBatchSize()));

		if (targetBatch.size() != m_Layers.back()->getSize().y * m_Layers.back()->getBatchSize())
			throw std::runtime_error("Wrong input_batch size! Should be: " + std::to_string(m_Layers.back()->getSize().y * m_Layers.back()->getBatchSize()));

		if (m_InputBuffer == 0 || m_TargetBuffer == 0) init();


		bindTrainingData(inputBatch, targetBatch);

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

		glGenBuffers(1, &m_InputBuffer);
		glGenBuffers(1, &m_TargetBuffer);

		glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_InputBuffer);
		glBufferData(GL_SHADER_STORAGE_BUFFER, m_Layers.front()->getBatchSize() * m_Layers.front()->getSize().x * sizeof(float), nullptr, GL_DYNAMIC_DRAW);

		glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_TargetBuffer);
		glBufferData(GL_SHADER_STORAGE_BUFFER, m_Layers.back()->getSize().y * m_Layers.back()->getBatchSize() * sizeof(float), nullptr, GL_DYNAMIC_DRAW);

		if (!m_ForwardPassCompute)	m_ForwardPassCompute = std::make_unique<Shader>("shaders/forward_pass.comp");
		if (!m_OutputDeltaCompute)	m_OutputDeltaCompute = std::make_unique<Shader>("shaders/delta_loss.comp");
		if (!m_HiddenDeltasCompute) m_HiddenDeltasCompute = std::make_unique<Shader>("shaders/backward_pass.comp");
		if (!m_WeightsCompute)		m_WeightsCompute = std::make_unique<Shader>("shaders/update_weights.comp");
		if (!m_BiasesCompute)		m_BiasesCompute = std::make_unique<Shader>("shaders/update_biases.comp");

	}
    
    // Interactive Testing CLI
	void NeuralNetwork::run() {

        std::cout << "\n=== Neural Network Testing Interface ===" << std::endl;
        std::cout << "Network trained to compute: sin(a) * sin(b)" << std::endl;
        std::cout << "Commands:" << std::endl;
        std::cout << "  test <a> <b>     - Test with specific values (angles in radians)" << std::endl;
        std::cout << "  random           - Test with random values" << std::endl;
        std::cout << "  batch <n>        - Test n random samples" << std::endl;
        std::cout << "  benchmark        - Performance benchmark" << std::endl;
        std::cout << "  quit             - Exit" << std::endl;
        std::cout << "  layer <i>        - Print layer" << std::endl;
        std::cout << "=====================================\n" << std::endl;

        std::string command;
        while (true) {
            std::cout << "nn> ";
            std::cin >> command;

            if (command == "quit" || command == "exit" || command == "q") {
                break;
            }
            else if (command == "test") {
                float a, b;
                if (std::cin >> a >> b) {
                    // Normalize inputs to [-1, 1] range
                    float inputs[2] = { a / 3.14159f, b / 3.14159f };
                    float expected = std::sin(a) * std::sin(b);

                    // Reuse forwardPass() for the actual computation
                    std::vector<float> inputVec(inputs, inputs + 2);
                    std::vector<float> targetVec(1);
                    bindTrainingData(inputVec, targetVec);
                    forwardPass();

                    // Read result
                    float result;
                    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_Layers.back()->m_ActivationBuffer);
                    float* mapped = (float*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
                    if (mapped) {
                        result = mapped[0];
                        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
                    }

                    float error = std::abs(result - expected);
                    float error_percent = (expected != 0) ? (error / std::abs(expected)) * 100.0f : error * 100.0f;

                    std::cout << std::fixed << std::setprecision(6);
                    std::cout << "Input: a=" << a << ", b=" << b << std::endl;
                    std::cout << "Expected: " << expected << std::endl;
                    std::cout << "Got:      " << result << std::endl;
                    std::cout << "Error:    " << error << " (" << error_percent << "%)" << std::endl;
                    std::cout << std::defaultfloat << std::endl;
                }
                else {
                    std::cout << "Usage: test <a> <b> (where a and b are angles in radians)" << std::endl;
                    std::cin.clear();
                    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                }
            }
            else if (command == "random") {
                float a = 3.14159f * ((float)rand() / RAND_MAX) * 2.0f - 3.14159f;
                float b = 3.14159f * ((float)rand() / RAND_MAX) * 2.0f - 3.14159f;

                float inputs[2] = { a / 3.14159f, b / 3.14159f };
                float expected = std::sin(a) * std::sin(b);

                std::cout << "Testing with random values: a=" << a << ", b=" << b << std::endl;

                // Reuse forwardPass() for the actual computation
                std::vector<float> inputVec(inputs, inputs + 2);
                std::vector<float> targetVec(1); 
                bindTrainingData(inputVec, targetVec);
                forwardPass();

                float result;
                glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_Layers.back()->m_ActivationBuffer);
                float* mapped = (float*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
                if (mapped) {
                    result = mapped[0];
                    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
                }

                float error = std::abs(result - expected);
                float error_percent = (expected != 0) ? (error / std::abs(expected)) * 100.0f : error * 100.0f;

                std::cout << std::fixed << std::setprecision(6);
                std::cout << "Expected: " << expected << std::endl;
                std::cout << "Got:      " << result << std::endl;
                std::cout << "Error:    " << error << " (" << error_percent << "%)" << std::endl;
                std::cout << std::defaultfloat << std::endl;
            }
            else if (command == "batch") {
                int n;
                if (std::cin >> n && n > 0 && n <= 1000) {
                    std::cout << "Testing " << n << " random samples..." << std::endl;

                    float total_error = 0.0f;
                    float max_error = 0.0f;
                    float min_error = std::numeric_limits<float>::max();

                    for (int i = 0; i < n; i++) {
                        float a = 3.14159f * ((float)rand() / RAND_MAX) * 2.0f - 3.14159f;
                        float b = 3.14159f * ((float)rand() / RAND_MAX) * 2.0f - 3.14159f;

                        float inputs[2] = { a / 3.14159f, b / 3.14159f };
                        float expected = std::sin(a) * std::sin(b);

                        // Reuse forwardPass() for the actual computation
                        std::vector<float> inputVec(inputs, inputs + 2);
                        std::vector<float> targetVec(1);
                        bindTrainingData(inputVec, targetVec);
                        forwardPass();

                        float result;
                        glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_Layers.back()->m_ActivationBuffer);
                        float* mapped = (float*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
                        if (mapped) {
                            result = mapped[0];
                            glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
                        }

                        float error = std::abs(result - expected);
                        total_error += error;
                        max_error = std::max(max_error, error);
                        min_error = std::min(min_error, error);
                    }

                    std::cout << std::fixed << std::setprecision(6);
                    std::cout << "Results for " << n << " samples:" << std::endl;
                    std::cout << "Average error: " << (total_error / n) << std::endl;
                    std::cout << "Min error:     " << min_error << std::endl;
                    std::cout << "Max error:     " << max_error << std::endl;
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
                    float a = 3.14159f * ((float)rand() / RAND_MAX) * 2.0f - 3.14159f;
                    float b = 3.14159f * ((float)rand() / RAND_MAX) * 2.0f - 3.14159f;
                    float inputs[2] = { a / 3.14159f, b / 3.14159f };

                    // Reuse forwardPass() for the actual computation
                    std::vector<float> inputVec(inputs, inputs + 2);
                    std::vector<float> targetVec(1);
                    bindTrainingData(inputVec, targetVec);
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
                std::cout << "  test <a> <b>     - Test with specific values (angles in radians)" << std::endl;
                std::cout << "  random           - Test with random values" << std::endl;
                std::cout << "  layer            - Show layer by id" << std::endl;
                std::cout << "  layer            - Show layer by id" << std::endl;
                std::cout << "  batch <n>        - Test n random samples (1-1000)" << std::endl;
                std::cout << "  benchmark        - Performance benchmark (10k samples)" << std::endl;
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
	}
}