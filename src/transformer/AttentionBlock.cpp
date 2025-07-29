#include "AttentionBlock.h"
#include "Logger.h"
#include <iostream>

namespace MLGL {

    AttentionBlock::AttentionBlock(int modelDimensions, int numHeads, int seqLen, bool mask)
        : m_ModelDim(modelDimensions), m_NumHeads(numHeads), m_HeadDim(modelDimensions / numHeads), m_SeqLen(seqLen), m_UseMask(mask), m_ADAM_Timestep(0) {

        // Validate that model dimension is divisible by number of heads
        if (modelDimensions % numHeads != 0) {
            throw std::runtime_error("Model dimension must be divisible by number of heads");
        }

        glGenBuffers(1, &m_PaddingMaskBuffer);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_PaddingMaskBuffer);
        glBufferData(GL_SHADER_STORAGE_BUFFER, seqLen * sizeof(int), nullptr, GL_STATIC_DRAW);

        // Initialize weight matrices
        m_WeightQueryMat = std::make_shared<Matrix>(modelDimensions, modelDimensions);
        m_WeightKeyMat = std::make_shared<Matrix>(modelDimensions, modelDimensions);
        m_WeightValueMat = std::make_shared<Matrix>(modelDimensions, modelDimensions);
        // Randomly initialize weights (using RELU initializer for consistency)
        for (int r = 0; r < modelDimensions; r++) {
            for (int c = 0; c < modelDimensions; c++) {
                m_WeightQueryMat->set(r, c, MLGL::activationFunctions[MLGL::ActivationFnType::RELU].weight_initializer(modelDimensions, modelDimensions));
                m_WeightKeyMat->set(r, c, MLGL::activationFunctions[MLGL::ActivationFnType::RELU].weight_initializer(modelDimensions, modelDimensions));
                m_WeightValueMat->set(r, c, MLGL::activationFunctions[MLGL::ActivationFnType::RELU].weight_initializer(modelDimensions, modelDimensions));
            }
        }
        m_WeightQueryMat->uploadToGPU();
        m_WeightKeyMat->uploadToGPU();
        m_WeightValueMat->uploadToGPU();

        // Initialize ADAM optimization buffers for Q, K, V weights
        m_ADAM_M_QueryMat = std::make_shared<Matrix>(modelDimensions, modelDimensions, 0.0f);
        m_ADAM_V_QueryMat = std::make_shared<Matrix>(modelDimensions, modelDimensions, 0.0f);
        m_ADAM_M_KeyMat = std::make_shared<Matrix>(modelDimensions, modelDimensions, 0.0f);
        m_ADAM_V_KeyMat = std::make_shared<Matrix>(modelDimensions, modelDimensions, 0.0f);
        m_ADAM_M_ValueMat = std::make_shared<Matrix>(modelDimensions, modelDimensions, 0.0f);
        m_ADAM_V_ValueMat = std::make_shared<Matrix>(modelDimensions, modelDimensions, 0.0f);

        // Initialize output matrices for Q, K, V projections
        m_OutQueryMat = std::make_shared<Matrix>(seqLen, modelDimensions);
        m_OutKeyMat = std::make_shared<Matrix>(seqLen, modelDimensions);
        m_OutValueMat = std::make_shared<Matrix>(seqLen, modelDimensions);

        // Initialize gradient matrices for Q, K, V projections
        m_GradQueryInputMat = std::make_shared<Matrix>(seqLen, modelDimensions, 0);
        m_GradKeyInputMat = std::make_shared<Matrix>(seqLen, modelDimensions, 0);
        m_GradValueInputMat = std::make_shared<Matrix>(seqLen, modelDimensions, 0);

        // Cached forward pass values for backprop
        m_CachedInput = nullptr;
        m_CachedContext = nullptr;

        m_CachedQ = std::make_shared<Matrix>(seqLen, modelDimensions);
        m_CachedK = std::make_shared<Matrix>(seqLen, modelDimensions);
        m_CachedV = std::make_shared<Matrix>(seqLen, modelDimensions);
        m_CachedQ->uploadToGPU();
        m_CachedK->uploadToGPU();
        m_CachedV->uploadToGPU();

        m_CachedScores = std::make_shared<Matrix>(numHeads * seqLen, seqLen);
        m_CachedScores->uploadToGPU();

        m_CachedAttentionWeights = std::make_shared<Matrix>(numHeads * seqLen, seqLen);
        m_CachedAttentionWeights->uploadToGPU();

        // Input gradients
        m_GradInput = std::make_shared<Matrix>(seqLen, modelDimensions, 0);
        m_GradContext = std::make_shared<Matrix>(seqLen, modelDimensions, 0);

        // Weight gradients
        m_GradWeightQueryMat = std::make_shared<Matrix>(modelDimensions, modelDimensions, 0);
        m_GradWeightKeyMat = std::make_shared<Matrix>(modelDimensions, modelDimensions, 0);
        m_GradWeightValueMat = std::make_shared<Matrix>(modelDimensions, modelDimensions, 0);

        // Intermediate gradients for backprop chain
        m_GradQ = std::make_shared<Matrix>(seqLen, modelDimensions, 0);
        m_GradK = std::make_shared<Matrix>(seqLen, modelDimensions, 0);
        m_GradV = std::make_shared<Matrix>(seqLen, modelDimensions, 0);
        m_GradScores = std::make_shared<Matrix>(numHeads * seqLen, seqLen, 0);
        m_GradAttentionWeights = std::make_shared<Matrix>(numHeads * seqLen, seqLen, 0);

        // Output matrix: [seq_len, model_dim] (concatenated heads)
        m_OutputMat = std::make_shared<Matrix>(seqLen, modelDimensions);
        m_OutputMat->uploadToGPU();

        loadShaders();
    }

    AttentionBlock::AttentionBlock(const char* data) : m_ADAM_Timestep(0) {
        if (!data) throw std::invalid_argument("Data pointer cannot be null");

        const char* ptr = data;

        // Read basic parameters
        std::memcpy(&m_ModelDim, ptr, sizeof(int));
        ptr += sizeof(int);

        std::memcpy(&m_NumHeads, ptr, sizeof(int));
        ptr += sizeof(int);

        std::memcpy(&m_HeadDim, ptr, sizeof(int));
        ptr += sizeof(int);

        std::memcpy(&m_SeqLen, ptr, sizeof(int));
        ptr += sizeof(int);

        std::memcpy(&m_UseMask, ptr, sizeof(bool));
        ptr += sizeof(bool);

        // Initialize weight matrices and load their data
        m_WeightQueryMat = std::make_shared<Matrix>(m_ModelDim, m_ModelDim, reinterpret_cast<const float*>(ptr));
        ptr += m_ModelDim * m_ModelDim * sizeof(float);
        m_WeightQueryMat->uploadToGPU();

        m_WeightKeyMat = std::make_shared<Matrix>(m_ModelDim, m_ModelDim, reinterpret_cast<const float*>(ptr));
        ptr += m_ModelDim * m_ModelDim * sizeof(float);
        m_WeightKeyMat->uploadToGPU();

        m_WeightValueMat = std::make_shared<Matrix>(m_ModelDim, m_ModelDim, reinterpret_cast<const float*>(ptr));
        ptr += m_ModelDim * m_ModelDim * sizeof(float);
        m_WeightValueMat->uploadToGPU();


        // Initialize OpenGL buffer for padding mask
        glGenBuffers(1, &m_PaddingMaskBuffer);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_PaddingMaskBuffer);
        glBufferData(GL_SHADER_STORAGE_BUFFER, m_SeqLen * sizeof(int), nullptr, GL_STATIC_DRAW);

        // Initialize ADAM optimization buffers for Q, K, V weights
        m_ADAM_M_QueryMat = std::make_shared<Matrix>(m_ModelDim, m_ModelDim, 0.0f);
        m_ADAM_V_QueryMat = std::make_shared<Matrix>(m_ModelDim, m_ModelDim, 0.0f);
        m_ADAM_M_KeyMat = std::make_shared<Matrix>(m_ModelDim, m_ModelDim, 0.0f);
        m_ADAM_V_KeyMat = std::make_shared<Matrix>(m_ModelDim, m_ModelDim, 0.0f);
        m_ADAM_M_ValueMat = std::make_shared<Matrix>(m_ModelDim, m_ModelDim, 0.0f);
        m_ADAM_V_ValueMat = std::make_shared<Matrix>(m_ModelDim, m_ModelDim, 0.0f);

        // Initialize output matrices for Q, K, V projections
        m_OutQueryMat = std::make_shared<Matrix>(m_SeqLen, m_ModelDim);
        m_OutKeyMat = std::make_shared<Matrix>(m_SeqLen, m_ModelDim);
        m_OutValueMat = std::make_shared<Matrix>(m_SeqLen, m_ModelDim);

        // Initialize gradient matrices for Q, K, V projections
        m_GradQueryInputMat = std::make_shared<Matrix>(m_SeqLen, m_ModelDim, 0);
        m_GradKeyInputMat = std::make_shared<Matrix>(m_SeqLen, m_ModelDim, 0);
        m_GradValueInputMat = std::make_shared<Matrix>(m_SeqLen, m_ModelDim, 0);

        // Cached forward pass values for backprop
        m_CachedInput = nullptr;
        m_CachedContext = nullptr;

        m_CachedQ = std::make_shared<Matrix>(m_SeqLen, m_ModelDim);
        m_CachedK = std::make_shared<Matrix>(m_SeqLen, m_ModelDim);
        m_CachedV = std::make_shared<Matrix>(m_SeqLen, m_ModelDim);
        m_CachedQ->uploadToGPU();
        m_CachedK->uploadToGPU();
        m_CachedV->uploadToGPU();

        m_CachedScores = std::make_shared<Matrix>(m_NumHeads * m_SeqLen, m_SeqLen);
        m_CachedScores->uploadToGPU();

        m_CachedAttentionWeights = std::make_shared<Matrix>(m_NumHeads * m_SeqLen, m_SeqLen);
        m_CachedAttentionWeights->uploadToGPU();

        // Input gradients
        m_GradInput = std::make_shared<Matrix>(m_SeqLen, m_ModelDim, 0);
        m_GradContext = std::make_shared<Matrix>(m_SeqLen, m_ModelDim, 0);

        // Weight gradients
        m_GradWeightQueryMat = std::make_shared<Matrix>(m_ModelDim, m_ModelDim, 0);
        m_GradWeightKeyMat = std::make_shared<Matrix>(m_ModelDim, m_ModelDim, 0);
        m_GradWeightValueMat = std::make_shared<Matrix>(m_ModelDim, m_ModelDim, 0);

        // Intermediate gradients for backprop chain
        m_GradQ = std::make_shared<Matrix>(m_SeqLen, m_ModelDim, 0);
        m_GradK = std::make_shared<Matrix>(m_SeqLen, m_ModelDim, 0);
        m_GradV = std::make_shared<Matrix>(m_SeqLen, m_ModelDim, 0);
        m_GradScores = std::make_shared<Matrix>(m_NumHeads * m_SeqLen, m_SeqLen, 0);
        m_GradAttentionWeights = std::make_shared<Matrix>(m_NumHeads * m_SeqLen, m_SeqLen, 0);

        // Output matrix: [seq_len, model_dim] (concatenated heads)
        m_OutputMat = std::make_shared<Matrix>(m_SeqLen, m_ModelDim);
        m_OutputMat->uploadToGPU();

        loadShaders();

        LOG_DEBUG("[ATTENTION LOAD] AttentionBlock loaded successfully from binary buffer");
    }

    int AttentionBlock::getSaveSize() {
        // Basic parameters: 5 ints + 1 bool
        size_t basic_params_size = 5 * sizeof(int) + sizeof(bool);

        // Weight matrices: 3 matrices of size [model_dim, model_dim]
        size_t weight_matrices_size = 3 * m_ModelDim * m_ModelDim * sizeof(float);

        return basic_params_size + weight_matrices_size;
    }

    const char* AttentionBlock::save() {
        // Allocate buffer (caller is responsible for freeing this memory)
        char* buffer = new char[getSaveSize()];
        char* ptr = buffer;

        // Save basic parameters
        std::memcpy(ptr, &m_ModelDim, sizeof(int));
        ptr += sizeof(int);

        std::memcpy(ptr, &m_NumHeads, sizeof(int));
        ptr += sizeof(int);

        std::memcpy(ptr, &m_HeadDim, sizeof(int));
        ptr += sizeof(int);

        std::memcpy(ptr, &m_SeqLen, sizeof(int));
        ptr += sizeof(int);

        std::memcpy(ptr, &m_UseMask, sizeof(bool));
        ptr += sizeof(bool);

        // Save weight matrices
        m_WeightQueryMat->downloadFromGPU();
        std::memcpy(ptr, m_WeightQueryMat->raw(), m_ModelDim * m_ModelDim * sizeof(float));
        ptr += m_ModelDim * m_ModelDim * sizeof(float);

        m_WeightKeyMat->downloadFromGPU();
        std::memcpy(ptr, m_WeightKeyMat->raw(), m_ModelDim * m_ModelDim * sizeof(float));
        ptr += m_ModelDim * m_ModelDim * sizeof(float);

        m_WeightValueMat->downloadFromGPU();
        std::memcpy(ptr, m_WeightValueMat->raw(), m_ModelDim * m_ModelDim * sizeof(float));
        ptr += m_ModelDim * m_ModelDim * sizeof(float);


        LOG_DEBUG("[ATTENTION SAVE] AttentionBlock saved successfully to binary buffer");

        return buffer;
    }

    void AttentionBlock::loadShaders() {
        // Load shaders
        m_ForwardPassWeightsCompute = ShaderManager::getInstance().getShader("shaders/attention/forward_weights.comp");
        m_ForwardPassScoreCompute = ShaderManager::getInstance().getShader("shaders/attention/forward_score.comp");
        m_SoftmaxCompute = ShaderManager::getInstance().getShader("shaders/attention/softmax.comp");
        m_ForwardPassOutCompute = ShaderManager::getInstance().getShader("shaders/attention/forward_output.comp");

        m_BackwardOutputCompute = ShaderManager::getInstance().getShader("shaders/attention/backward_output.comp");
        m_BackwardScoresCompute = ShaderManager::getInstance().getShader("shaders/attention/backward_scores.comp");
        m_WeightsUpdatePassCompute = ShaderManager::getInstance().getShader("shaders/attention/weights_update_adam.comp");
        m_BackwardProjectionsCompute = ShaderManager::getInstance().getShader("shaders/attention/backward_projections.comp");

        m_GradInputCompute = ShaderManager::getInstance().getShader("shaders/attention/backward_grad_input.comp");
        m_GradWeightCompute = ShaderManager::getInstance().getShader("shaders/attention/backward_grad_weight.comp");
    }

    std::shared_ptr<Matrix> AttentionBlock::forward(const std::shared_ptr<Matrix>& input, const std::shared_ptr<Matrix>& context) {
        MLGL::Timer timer("AttentionBlock::forward");
        // CACHE INPUT FOR BACKPROP
        m_CachedInput = input;
        if (context) m_CachedContext = context;

        const auto& input_kv = context ? context : input;

        if (!m_PaddingMask.empty()) {
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_PaddingMaskBuffer);
            glBufferData(GL_SHADER_STORAGE_BUFFER, m_PaddingMask.size() * sizeof(int), m_PaddingMask.data(), GL_STATIC_DRAW);
        }

        // === STEP 1: Compute Q, K, V projections ===
        {
            m_ForwardPassWeightsCompute->bindBuffer(0, "InputQ", input->buffer);
            m_ForwardPassWeightsCompute->bindBuffer(1, "InputKV", input_kv->buffer);
            m_ForwardPassWeightsCompute->bindBuffer(2, "WeightQ", m_WeightQueryMat->buffer);
            m_ForwardPassWeightsCompute->bindBuffer(3, "WeightK", m_WeightKeyMat->buffer);
            m_ForwardPassWeightsCompute->bindBuffer(4, "WeightV", m_WeightValueMat->buffer);

            // CACHE Q, K, V for backprop
            m_ForwardPassWeightsCompute->bindBuffer(5, "OutputQ", m_CachedQ->buffer);
            m_ForwardPassWeightsCompute->bindBuffer(6, "OutputK", m_CachedK->buffer);
            m_ForwardPassWeightsCompute->bindBuffer(7, "OutputV", m_CachedV->buffer);

            m_ForwardPassWeightsCompute->setUniform("model_dim", m_ModelDim);
            m_ForwardPassWeightsCompute->setUniform("input_dim", input->cols);
            m_ForwardPassWeightsCompute->setUniform("seq_len", input->rows);

            int workgroups_x = (input->rows * m_ModelDim + 31) / 32;
            m_ForwardPassWeightsCompute->dispatch(workgroups_x, 1, 1);

            for (int i = 0; i <= 7; i++) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, 0);
        }

        // === STEP 2: Compute attention scores (Q * K^T / sqrt(d_k)) ===
        {
            // This computes scores = Q @ K^T / sqrt(head_dim) for each head
            // CACHE the raw scores BEFORE softmax
            m_ForwardPassScoreCompute->bindBuffer(0, "BufferQ", m_CachedQ->buffer);
            m_ForwardPassScoreCompute->bindBuffer(1, "BufferK", m_CachedK->buffer);
            m_ForwardPassScoreCompute->bindBuffer(2, "RawScores", m_CachedScores->buffer);  // CACHE THIS

            m_ForwardPassScoreCompute->setUniform("has_padding_mask", !m_PaddingMask.empty());
            if (!m_PaddingMask.empty()) m_ForwardPassScoreCompute->bindBuffer(3, "PaddingMask", m_PaddingMaskBuffer); // Bind padding mask if available

            m_ForwardPassScoreCompute->setUniform("seq_len", input->rows);
            m_ForwardPassScoreCompute->setUniform("head_dim", m_HeadDim);
            m_ForwardPassScoreCompute->setUniform("num_heads", m_NumHeads);
            m_ForwardPassScoreCompute->setUniform("use_mask", m_UseMask ? 1 : 0);
            float invSqrtHeadDim = 1.0f / std::sqrt(static_cast<float>(m_HeadDim));
            m_ForwardPassScoreCompute->setUniform("inv_sqrt_head_dim", invSqrtHeadDim);

            int workgroups_x = (input->rows + 15) / 16;
            int workgroups_y = (input->rows + 15) / 16;  // seq_len x seq_len
            m_ForwardPassScoreCompute->dispatch(workgroups_x, workgroups_y, 1);

            for (int i = 0; i <= 3; ++i) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, 0);
        }

        // === STEP 3: Apply softmax to get attention weights ===
        {
            // Apply softmax to cached scores, store in cached attention weights
            m_SoftmaxCompute->bindBuffer(0, "Input", m_CachedScores->buffer);
            m_SoftmaxCompute->bindBuffer(1, "Output", m_CachedAttentionWeights->buffer);  // CACHE THIS

            m_SoftmaxCompute->setUniform("has_padding_mask", !m_PaddingMask.empty());
            if (!m_PaddingMask.empty()) m_SoftmaxCompute->bindBuffer(2, "PaddingMask", m_PaddingMaskBuffer); // Bind padding mask if available

            m_SoftmaxCompute->setUniform("seq_len", input->rows);
            m_SoftmaxCompute->setUniform("num_heads", m_NumHeads);
            m_SoftmaxCompute->setUniform("use_mask", m_UseMask ? 1 : 0);

            int total_rows = m_NumHeads * input->rows;
            int workgroups = (total_rows + 15) / 16;
            m_SoftmaxCompute->dispatch(workgroups, 1, 1);

            for (int i = 0; i <= 2; ++i) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, 0);
        }

        // === STEP 4: Compute final output (attention_weights @ V) ===
        {
            // Final matrix multiply: attention_weights @ V -> output
            m_ForwardPassOutCompute->bindBuffer(0, "AttentionWeights", m_CachedAttentionWeights->buffer);
            m_ForwardPassOutCompute->bindBuffer(1, "BufferV", m_CachedV->buffer);
            m_ForwardPassOutCompute->bindBuffer(2, "BufferOutput", m_OutputMat->buffer);

            m_ForwardPassOutCompute->setUniform("seq_len", input->rows);
            m_ForwardPassOutCompute->setUniform("head_dim", m_HeadDim);
            m_ForwardPassOutCompute->setUniform("num_heads", m_NumHeads);

            int workgroups_x = (input->rows + 15) / 16;
            int workgroups_y = (m_ModelDim + 15) / 16;
            m_ForwardPassOutCompute->dispatch(workgroups_x, workgroups_y, 1);
            
            for (int i = 0; i <= 2; ++i) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, 0);
        }
        //we don't download data we leave them in gpu so no need to worry that array is empty
        return m_OutputMat;
    }

    std::shared_ptr<Matrix> AttentionBlock::forward(const std::shared_ptr<Matrix>& input, const std::shared_ptr<Matrix>& input_kv, const std::vector<int>& paddingMask) {
        MLGL::Timer timer("AttentionBlock::forward (with mask)");
        // Store padding mask for use in shaders
        m_PaddingMask = paddingMask;
        
        // Use the existing forward implementation
        return forward(input, input_kv);
    }
 
    std::pair<std::shared_ptr<Matrix>, std::shared_ptr<Matrix>> AttentionBlock::backward( const std::shared_ptr<Matrix>& gradOutput, const std::shared_ptr<Matrix>& context, float learningRate ) {
        MLGL::Timer timer("AttentionBlock::backward");
        
        if (!m_PaddingMask.empty()) {
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_PaddingMaskBuffer);
            glBufferData(GL_SHADER_STORAGE_BUFFER, m_PaddingMask.size() * sizeof(int), m_PaddingMask.data(), GL_STATIC_DRAW);
        }

        // === STEP 1: Backward through final matmul (grad_output -> grad_attention_weights, grad_V) ===
        { 
            m_BackwardOutputCompute->bindBuffer(0, "GradOutput", gradOutput->buffer);
            m_BackwardOutputCompute->bindBuffer(1, "CachedV", m_CachedV->buffer);
            m_BackwardOutputCompute->bindBuffer(2, "CachedAttentionWeights", m_CachedAttentionWeights->buffer);
            m_BackwardOutputCompute->bindBuffer(3, "GradAttentionWeights", m_GradAttentionWeights->buffer);
            m_BackwardOutputCompute->bindBuffer(4, "GradV", m_GradV->buffer);

            m_BackwardOutputCompute->setUniform("seq_len", m_SeqLen);
            m_BackwardOutputCompute->setUniform("head_dim", m_HeadDim);

            int workgroups_x = (m_SeqLen + 15) / 16;
            int workgroups_y = (m_SeqLen + 15) / 16;
            m_BackwardOutputCompute->dispatch(workgroups_x, workgroups_y, 1);
            for (int i = 0; i <= 4; ++i) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, 0);
        }

        // === STEP 2: Backward through softmax ===
        {
            // This is: grad_scores[i,j] = attention_weights[i,j] * (grad_attention_weights[i,j] - sum_k(grad_attention_weights[i,k] * attention_weights[i,k]))
            m_BackwardScoresCompute->bindBuffer(0, "GradAttentionWeights", m_GradAttentionWeights->buffer);
            m_BackwardScoresCompute->bindBuffer(1, "CachedAttentionWeights", m_CachedAttentionWeights->buffer);
            m_BackwardScoresCompute->bindBuffer(2, "GradScores", m_GradScores->buffer);

            m_BackwardScoresCompute->setUniform("has_padding_mask", !m_PaddingMask.empty());
            if (!m_PaddingMask.empty())m_BackwardScoresCompute->bindBuffer(3, "PaddingMask", m_PaddingMaskBuffer); // Bind padding mask if available
            m_BackwardScoresCompute->setUniform("seq_len", m_SeqLen);
            m_BackwardScoresCompute->setUniform("use_mask", m_UseMask ? 1 : 0);

            int workgroups_x = (m_SeqLen + 15) / 16;
            m_BackwardScoresCompute->dispatch(workgroups_x, 1, 1);
            for (int i = 0; i <= 3; ++i) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, 0);
        }

        // === STEP 3: Backward through scores computation (Q @ K^T / sqrt(d_k)) ===
        {
            m_BackwardProjectionsCompute->bindBuffer(0, "GradScores", m_GradScores->buffer);
            m_BackwardProjectionsCompute->bindBuffer(1, "CachedQ", m_CachedQ->buffer);
            m_BackwardProjectionsCompute->bindBuffer(2, "CachedK", m_CachedK->buffer);
            m_BackwardProjectionsCompute->bindBuffer(3, "GradQ", m_GradQ->buffer);
            m_BackwardProjectionsCompute->bindBuffer(4, "GradK", m_GradK->buffer);

            m_BackwardProjectionsCompute->setUniform("seq_len", m_SeqLen);
            m_BackwardProjectionsCompute->setUniform("head_dim", m_HeadDim);
            float invSqrtHeadDim = 1.0f / std::sqrt(static_cast<float>(m_HeadDim));
            m_BackwardProjectionsCompute->setUniform("inv_sqrt_head_dim", invSqrtHeadDim);

            int workgroups_x = (m_SeqLen * m_ModelDim + 31) / 32;
            m_BackwardProjectionsCompute->dispatch(workgroups_x, 1, 1);
            for (int i = 0; i <= 4; ++i) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, 0);
        }

        // === STEP 4: Backward through linear projections ===
        const auto& keyValueInput = context ? m_CachedContext : m_CachedInput;

        // Accumulate Q, K, V gradients directly into m_GradInput/m_GradContext
        computeProjectionGradients(m_GradQ, m_CachedInput, m_WeightQueryMat, m_GradInput, m_GradWeightQueryMat, /*accumulate=*/false);
        if (context) {
            computeProjectionGradients(m_GradK, keyValueInput, m_WeightKeyMat, m_GradContext, m_GradWeightKeyMat, /*accumulate=*/false);
            computeProjectionGradients(m_GradV, keyValueInput, m_WeightValueMat, m_GradContext, m_GradWeightValueMat, /*accumulate=*/true);
        }
        else {
            computeProjectionGradients(m_GradK, keyValueInput, m_WeightKeyMat, m_GradInput, m_GradWeightKeyMat, /*accumulate=*/true);
            computeProjectionGradients(m_GradV, keyValueInput, m_WeightValueMat, m_GradInput, m_GradWeightValueMat, /*accumulate=*/true);
        }

        // === STEP 5: Update weights with ADAM ===
        updateWeights(m_WeightQueryMat, m_GradWeightQueryMat, m_ADAM_M_QueryMat, m_ADAM_V_QueryMat, learningRate);
        updateWeights(m_WeightKeyMat, m_GradWeightKeyMat, m_ADAM_M_KeyMat, m_ADAM_V_KeyMat, learningRate);
        updateWeights(m_WeightValueMat, m_GradWeightValueMat, m_ADAM_M_ValueMat, m_ADAM_V_ValueMat, learningRate);

        // Increment ADAM timestep
        m_ADAM_Timestep++;
        return std::make_pair(m_GradInput, m_GradContext);
    }

    void AttentionBlock::updateWeights(const std::shared_ptr<Matrix>& weight, const std::shared_ptr<Matrix>& gradWeight, 
                                     const std::shared_ptr<Matrix>& adamM, const std::shared_ptr<Matrix>& adamV, float learningRate) {

        m_WeightsUpdatePassCompute->bindBuffer(0, "Weight", weight->buffer);
        m_WeightsUpdatePassCompute->bindBuffer(1, "GradWeight", gradWeight->buffer);
        m_WeightsUpdatePassCompute->bindBuffer(2, "ADAM_M", adamM->buffer);
        m_WeightsUpdatePassCompute->bindBuffer(3, "ADAM_V", adamV->buffer);

        m_WeightsUpdatePassCompute->setUniform("input_dim", weight->rows);
        m_WeightsUpdatePassCompute->setUniform("head_dim", weight->cols);
        m_WeightsUpdatePassCompute->setUniform("learning_rate", learningRate);
        m_WeightsUpdatePassCompute->setUniform("ADAM_beta1", 0.9f);
        m_WeightsUpdatePassCompute->setUniform("ADAM_beta2", 0.999f);
        m_WeightsUpdatePassCompute->setUniform("ADAM_timestep", m_ADAM_Timestep);

        int workgroups_x = (weight->rows * weight->cols + 31) / 32;
        m_WeightsUpdatePassCompute->dispatch(workgroups_x, 1, 1);

        for (int i = 0; i <= 3; i++) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, 0);
    }

    void AttentionBlock::computeProjectionGradients(const std::shared_ptr<Matrix>& gradProjection, const std::shared_ptr<Matrix>& cachedInput, const std::shared_ptr<Matrix>& weight,
        std::shared_ptr<Matrix>& gradInput, std::shared_ptr<Matrix>& gradWeight, bool accumulate) {
        // grad_input = grad_projection @ W^T
        // grad_weight = input^T @ grad_projection
            {
                m_GradInputCompute->bindBuffer(0, "GradProjection", gradProjection->buffer);
                m_GradInputCompute->bindBuffer(1, "Weight", weight->buffer);
                m_GradInputCompute->bindBuffer(2, "GradInput", gradInput->buffer);

                m_GradInputCompute->setUniform("seq_len", m_SeqLen);
                m_GradInputCompute->setUniform("input_dim", m_ModelDim);
                m_GradInputCompute->setUniform("head_dim", m_HeadDim);
                m_GradInputCompute->setUniform("accumulate", accumulate ? 1 : 0); // Pass accumulate flag

                // Dispatch with appropriate work group sizes
                int workgroups_x = (m_ModelDim + 15) / 16;
                int workgroups_y = (m_SeqLen + 15) / 16;
                m_GradInputCompute->dispatch(workgroups_x, workgroups_y, 1);

                for (int i = 0; i <= 2; ++i) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, 0);
            }

            // === Compute grad_weight = input^T @ grad_projection ===
            {
                m_GradWeightCompute->bindBuffer(0, "CachedInput", cachedInput->buffer);
                m_GradWeightCompute->bindBuffer(1, "GradProjection", gradProjection->buffer);
                m_GradWeightCompute->bindBuffer(2, "GradWeight", gradWeight->buffer);

                m_GradWeightCompute->setUniform("seq_len", m_SeqLen);
                m_GradWeightCompute->setUniform("input_dim", m_ModelDim);
                m_GradWeightCompute->setUniform("head_dim", m_HeadDim);
                m_GradWeightCompute->setUniform("accumulate", accumulate ? 1 : 0); // Pass accumulate flag

                // Debug: print buffer size and dispatch dimensions
                int workgroups_x = (m_ModelDim + 15) / 16;
                int workgroups_y = (m_ModelDim + 15) / 16; // Changed from m_HeadDim to m_ModelDim

                // Dispatch with appropriate work group sizes
                m_GradWeightCompute->dispatch(workgroups_x, workgroups_y, 1);

                for (int i = 0; i <= 2; ++i) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, 0);
            }
    }
}
