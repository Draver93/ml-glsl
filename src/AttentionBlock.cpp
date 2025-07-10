#include "AttentionBlock.h"
#include "Logger.h"

namespace NNGL {
    AttentionBlock::AttentionBlock(int modelDimensions, int numHeads, int seqLen, bool mask)
        : m_ModelDim(modelDimensions), m_NumHeads(numHeads), m_HeadDim(modelDimensions / numHeads), m_SeqLen(seqLen), m_UseMask(mask), m_ADAM_Timestep(0) {

        // Validate that model dimension is divisible by number of heads
        if (modelDimensions % numHeads != 0) {
            throw std::runtime_error("Model dimension must be divisible by number of heads");
        }

        // Initialize weight matrices
        m_WeightQueryMat = std::make_shared<Matrix>(modelDimensions, modelDimensions);
        m_WeightKeyMat = std::make_shared<Matrix>(modelDimensions, modelDimensions);
        m_WeightValueMat = std::make_shared<Matrix>(modelDimensions, modelDimensions);
        // Randomly initialize weights (using RELU initializer for consistency)
        for (int r = 0; r < modelDimensions; r++) {
            for (int c = 0; c < modelDimensions; c++) {
                m_WeightQueryMat->set(r, c, NNGL::activationFunctions[NNGL::ActivationFnType::RELU].weight_initializer(modelDimensions, modelDimensions));
                m_WeightKeyMat->set(r, c, NNGL::activationFunctions[NNGL::ActivationFnType::RELU].weight_initializer(modelDimensions, modelDimensions));
                m_WeightValueMat->set(r, c, NNGL::activationFunctions[NNGL::ActivationFnType::RELU].weight_initializer(modelDimensions, modelDimensions));
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
        m_CachedInput = std::make_shared<Matrix>(seqLen, modelDimensions);
        m_CachedContext = std::make_shared<Matrix>(seqLen, modelDimensions);
        m_CachedQ = std::make_shared<Matrix>(seqLen, modelDimensions);
        m_CachedK = std::make_shared<Matrix>(seqLen, modelDimensions);
        m_CachedV = std::make_shared<Matrix>(seqLen, modelDimensions);
        m_CachedScores = std::make_shared<Matrix>(numHeads * seqLen, seqLen);
        m_CachedAttentionWeights = std::make_shared<Matrix>(numHeads * seqLen, seqLen);

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
        NNGL::Timer timer("AttentionBlock::forward");
        // CACHE INPUT FOR BACKPROP
        m_CachedInput->copyFrom(input);
        if (context) {
            if (!m_CachedContext) {
                m_CachedContext = std::make_shared<Matrix>(context->rows, context->cols);
            }
            m_CachedContext->copyFrom(context);
        }

        const auto& input_kv = context ? context : input;

        // === STEP 1: Compute Q, K, V projections ===
        {
            input->uploadToGPU();
            input_kv->uploadToGPU();

            m_ForwardPassWeightsCompute->bindBuffer(0, "InputQ", input->buffer);
            m_ForwardPassWeightsCompute->bindBuffer(1, "InputKV", input_kv->buffer);
            m_ForwardPassWeightsCompute->bindBuffer(2, "WeightQ", m_WeightQueryMat->buffer);
            m_ForwardPassWeightsCompute->bindBuffer(3, "WeightK", m_WeightKeyMat->buffer);
            m_ForwardPassWeightsCompute->bindBuffer(4, "WeightV", m_WeightValueMat->buffer);

            // CACHE Q, K, V for backprop
            m_ForwardPassWeightsCompute->bindBuffer(5, "OutputQ", m_CachedQ->buffer);
            m_ForwardPassWeightsCompute->bindBuffer(6, "OutputK", m_CachedK->buffer);
            m_ForwardPassWeightsCompute->bindBuffer(7, "OutputV", m_CachedV->buffer);

            m_ForwardPassWeightsCompute->setUniform("head_dim", m_HeadDim);
            m_ForwardPassWeightsCompute->setUniform("num_heads", m_NumHeads);
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
            
            // Bind padding mask if available
            if (!m_PaddingMask.empty()) {
                // Create a buffer for the padding mask
                GLuint paddingMaskBuffer;
                glGenBuffers(1, &paddingMaskBuffer);
                glBindBuffer(GL_SHADER_STORAGE_BUFFER, paddingMaskBuffer);
                glBufferData(GL_SHADER_STORAGE_BUFFER, m_PaddingMask.size() * sizeof(int), m_PaddingMask.data(), GL_STATIC_DRAW);
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, paddingMaskBuffer);
                m_ForwardPassScoreCompute->bindBuffer(3, "PaddingMask", paddingMaskBuffer);
            }

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
            
            // Bind padding mask if available
            if (!m_PaddingMask.empty()) {
                // Create a buffer for the padding mask
                GLuint paddingMaskBuffer;
                glGenBuffers(1, &paddingMaskBuffer);
                glBindBuffer(GL_SHADER_STORAGE_BUFFER, paddingMaskBuffer);
                glBufferData(GL_SHADER_STORAGE_BUFFER, m_PaddingMask.size() * sizeof(int), m_PaddingMask.data(), GL_STATIC_DRAW);
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, paddingMaskBuffer);
                m_SoftmaxCompute->bindBuffer(2, "PaddingMask", paddingMaskBuffer);
            }

            m_SoftmaxCompute->setUniform("seq_len", input->rows);
            m_SoftmaxCompute->setUniform("num_heads", m_NumHeads);
            m_SoftmaxCompute->setUniform("use_mask", m_UseMask ? 1 : 0);

            int workgroups = (input->rows + 15) / 16;
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
        NNGL::Timer timer("AttentionBlock::forward (with mask)");
        // Store padding mask for use in shaders
        m_PaddingMask = paddingMask;
        
        // Use the existing forward implementation
        return forward(input, input_kv);
    }
 
    std::pair<std::shared_ptr<Matrix>, std::shared_ptr<Matrix>> AttentionBlock::backward( const std::shared_ptr<Matrix>& gradOutput, const std::shared_ptr<Matrix>& input, const std::shared_ptr<Matrix>& context ) {
        NNGL::Timer timer("AttentionBlock::backward");
        gradOutput->uploadToGPU();
        const int seqLen = input->rows;
        const int inputDim = input->cols;
        const int headDim = m_HeadDim;

        // === STEP 1: Backward through final matmul (grad_output -> grad_attention_weights, grad_V) ===
        {
            // grad_attention_weights = grad_output @ V^T
            // grad_V = attention_weights^T @ grad_output

            m_BackwardOutputCompute->bindBuffer(0, "GradOutput", gradOutput->buffer);
            m_BackwardOutputCompute->bindBuffer(1, "CachedV", m_CachedV->buffer);
            m_BackwardOutputCompute->bindBuffer(2, "CachedAttentionWeights", m_CachedAttentionWeights->buffer);
            m_BackwardOutputCompute->bindBuffer(3, "GradAttentionWeights", m_GradAttentionWeights->buffer);
            m_BackwardOutputCompute->bindBuffer(4, "GradV", m_GradV->buffer);

            m_BackwardOutputCompute->setUniform("seq_len", seqLen);
            m_BackwardOutputCompute->setUniform("head_dim", headDim);
            m_BackwardOutputCompute->setUniform("num_heads", m_NumHeads);

            int workgroups_x = (seqLen + 15) / 16;
            int workgroups_y = (seqLen + 15) / 16;
            m_BackwardOutputCompute->dispatch(workgroups_x, workgroups_y, 1);

            for (int i = 0; i <= 4; ++i) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, 0);
        }

        // === STEP 2: Backward through softmax ===
        {
            // grad_scores = softmax_backward(grad_attention_weights, cached_attention_weights)
            // This is: grad_scores[i,j] = attention_weights[i,j] * (grad_attention_weights[i,j] - sum_k(grad_attention_weights[i,k] * attention_weights[i,k]))

            m_BackwardScoresCompute->bindBuffer(0, "GradAttentionWeights", m_GradAttentionWeights->buffer);
            m_BackwardScoresCompute->bindBuffer(1, "CachedAttentionWeights", m_CachedAttentionWeights->buffer);
            m_BackwardScoresCompute->bindBuffer(2, "GradScores", m_GradScores->buffer);
            
            // Bind padding mask if available
            if (!m_PaddingMask.empty()) {
                // Create a buffer for the padding mask
                GLuint paddingMaskBuffer;
                glGenBuffers(1, &paddingMaskBuffer);
                glBindBuffer(GL_SHADER_STORAGE_BUFFER, paddingMaskBuffer);
                glBufferData(GL_SHADER_STORAGE_BUFFER, m_PaddingMask.size() * sizeof(int), m_PaddingMask.data(), GL_STATIC_DRAW);
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, paddingMaskBuffer);
                m_BackwardScoresCompute->bindBuffer(3, "PaddingMask", paddingMaskBuffer);
            }

            m_BackwardScoresCompute->setUniform("seq_len", seqLen);
            m_BackwardScoresCompute->setUniform("num_heads", m_NumHeads);
            m_BackwardScoresCompute->setUniform("use_mask", m_UseMask ? 1 : 0);

            int workgroups_x = (seqLen + 15) / 16;
            m_BackwardScoresCompute->dispatch(workgroups_x, 1, 1);

            for (int i = 0; i <= 3; ++i) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, 0);
        }

        // === STEP 3: Backward through scores computation (Q @ K^T / sqrt(d_k)) ===
        {
            // grad_Q = grad_scores @ K / sqrt(d_k)
            // grad_K = grad_scores^T @ Q / sqrt(d_k)

            m_BackwardProjectionsCompute->bindBuffer(0, "GradScores", m_GradScores->buffer);
            m_BackwardProjectionsCompute->bindBuffer(1, "CachedQ", m_CachedQ->buffer);
            m_BackwardProjectionsCompute->bindBuffer(2, "CachedK", m_CachedK->buffer);
            m_BackwardProjectionsCompute->bindBuffer(3, "GradQ", m_GradQ->buffer);
            m_BackwardProjectionsCompute->bindBuffer(4, "GradK", m_GradK->buffer);

            m_BackwardProjectionsCompute->setUniform("seq_len", seqLen);
            m_BackwardProjectionsCompute->setUniform("head_dim", headDim);
            m_BackwardProjectionsCompute->setUniform("num_heads", m_NumHeads);
            float invSqrtHeadDim = 1.0f / std::sqrt(static_cast<float>(headDim));
            m_BackwardProjectionsCompute->setUniform("inv_sqrt_head_dim", invSqrtHeadDim);

            int workgroups_x = (seqLen * m_ModelDim + 31) / 32;
            m_BackwardProjectionsCompute->dispatch(workgroups_x, 1, 1);

            for (int i = 0; i <= 4; ++i) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, 0);
        }

        // === STEP 4: Backward through linear projections ===
        const auto& keyValueInput = context ? m_CachedContext : m_CachedInput;

        // Query gradients (always from input)
        computeProjectionGradients(m_GradQ, m_CachedInput, m_WeightQueryMat,
            m_GradInput, m_GradWeightQueryMat);

        // Key gradients (from input or context)
        if (context) {
            computeProjectionGradients(m_GradK, keyValueInput, m_WeightKeyMat,
                m_GradContext, m_GradWeightKeyMat);
        }
        else {
            auto tempGradInput = std::make_shared<Matrix>(seqLen, inputDim, 0);
            computeProjectionGradients(m_GradK, keyValueInput, m_WeightKeyMat,
                tempGradInput, m_GradWeightKeyMat);
            m_GradInput->add(*tempGradInput);
        }

        // Value gradients (from input or context)
        if (context) {
            auto tempGradContext = std::make_shared<Matrix>(seqLen, inputDim, 0);
            computeProjectionGradients(m_GradV, keyValueInput, m_WeightValueMat,
                tempGradContext, m_GradWeightValueMat);
            m_GradContext->add(*tempGradContext);
        }
        else {
            auto tempGradInput = std::make_shared<Matrix>(seqLen, inputDim, 0);
            computeProjectionGradients(m_GradV, keyValueInput, m_WeightValueMat,
                tempGradInput, m_GradWeightValueMat);
            m_GradInput->add(*tempGradInput);
        }

        // === STEP 5: Update weights with ADAM ===
        updateWeights(m_WeightQueryMat, m_GradWeightQueryMat, m_ADAM_M_QueryMat, m_ADAM_V_QueryMat, 0.001f);
        updateWeights(m_WeightKeyMat, m_GradWeightKeyMat, m_ADAM_M_KeyMat, m_ADAM_V_KeyMat, 0.001f);
        updateWeights(m_WeightValueMat, m_GradWeightValueMat, m_ADAM_M_ValueMat, m_ADAM_V_ValueMat, 0.001f);

        // Increment ADAM timestep
        m_ADAM_Timestep++;

        // Return gradient w.r.t. input
        return std::make_pair(m_GradInput, m_GradContext);
    }

    void AttentionBlock::updateWeights(const std::shared_ptr<Matrix>& weight, const std::shared_ptr<Matrix>& gradWeight, 
                                     const std::shared_ptr<Matrix>& adamM, const std::shared_ptr<Matrix>& adamV, float learningRate) {

        weight->uploadToGPU();
        gradWeight->uploadToGPU();
        adamM->uploadToGPU();
        adamV->uploadToGPU();

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

    void AttentionBlock::computeProjectionGradients(const std::shared_ptr<Matrix>& gradProjection,
        const std::shared_ptr<Matrix>& cachedInput, const std::shared_ptr<Matrix>& weight,
        std::shared_ptr<Matrix>& gradInput, std::shared_ptr<Matrix>& gradWeight) {
        // grad_input = grad_projection @ W^T
        // grad_weight = input^T @ grad_projection

        gradProjection->uploadToGPU();
        cachedInput->uploadToGPU();
        weight->uploadToGPU();

        // Compute gradients using your existing shaders or new ones
        // This is similar to your existing gradient computation but using cached values

        const int seq_len = cachedInput->rows;
        const int input_dim = cachedInput->cols;
        const int head_dim = weight->cols;

        // Reset gradients
        gradInput->clear(0.0f);
        gradWeight->clear(0.0f);

        {
            m_GradInputCompute->bindBuffer(0, "GradProjection", gradProjection->buffer);
            m_GradInputCompute->bindBuffer(1, "Weight", weight->buffer);
            m_GradInputCompute->bindBuffer(2, "GradInput", gradInput->buffer);

            m_GradInputCompute->setUniform("seq_len", seq_len);
            m_GradInputCompute->setUniform("input_dim", input_dim);
            m_GradInputCompute->setUniform("head_dim", head_dim);

            // Dispatch with appropriate work group sizes
            int workgroups_x = (seq_len + 15) / 16;
            int workgroups_y = (input_dim + 15) / 16;
            m_GradInputCompute->dispatch(workgroups_x, workgroups_y, 1);

            for (int i = 0; i <= 2; ++i) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, 0);
        }

        // === Compute grad_weight = input^T @ grad_projection ===
        {
            m_GradWeightCompute->bindBuffer(0, "CachedInput", cachedInput->buffer);
            m_GradWeightCompute->bindBuffer(1, "GradProjection", gradProjection->buffer);
            m_GradWeightCompute->bindBuffer(2, "GradWeight", gradWeight->buffer);

            m_GradWeightCompute->setUniform("seq_len", seq_len);
            m_GradWeightCompute->setUniform("input_dim", input_dim);
            m_GradWeightCompute->setUniform("head_dim", head_dim);

            // Dispatch with appropriate work group sizes
            int workgroups_x = (input_dim + 15) / 16;
            int workgroups_y = (head_dim + 15) / 16;
            m_GradWeightCompute->dispatch(workgroups_x, workgroups_y, 1);

            for (int i = 0; i <= 2; ++i) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, 0);
        }
    }
}
