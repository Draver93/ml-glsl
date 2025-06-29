#pragma once

#include "EmbeddingBlock.h"
#include "EncoderBlock.h"
#include "DecoderBlock.h"
#include "BPE.h"
#include "NeuralNetwork.h"
#include "Matrix.h"

#include <memory>
#include <string>
#include <vector>

namespace NNGL {
    class Transformer {
    private:
        std::unique_ptr<BPE> m_Tokenizer;
        std::unique_ptr<EmbeddingBlock> m_Embedder;
        std::unique_ptr<EncoderBlock> m_Encoder;
        std::unique_ptr<DecoderBlock> m_Decoder;
        std::unique_ptr<NeuralNetwork> m_OutputProjection;  // W_out as NN layer

        size_t m_SeqLen, m_VocabSize;
        size_t m_trainStep = 0;

    public:
        Transformer(std::string tokCheckpointFilepath, int modelDim, int hiddenDim, int seqLen);
        
        void train(const std::string& inputText);
        void trainOnSequence(const std::vector<std::string>& longSequence, size_t windowSize = 0, float learningRate = 0.001f);
        void trainNextToken(const std::vector<std::string>& inputTokens, float learningRate = 0.001f);
        std::string eval(std::string& inputText);

    private:
        std::shared_ptr<Matrix> forwardPass(std::vector<std::string>& encInputTokens, std::vector<std::string>& decInputTokens);
        void backwardPass(const std::vector<std::string>& encInputTokens,
            const std::vector<std::string>& decInputTokens,
            std::shared_ptr<Matrix> targetMat,
            float learningRate);
        int predictToken(std::shared_ptr<Matrix> logits);
        void printGradientHeatmap(std::shared_ptr<Matrix> mat);
    };
}