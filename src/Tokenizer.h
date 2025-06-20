#pragma once
#include <iostream>
#include <random>
#include <unordered_map>
#include "Matrix.h"
#include "BPE.h"

/*namespace NNGL {

    class TokenVectorMapper {
    public:
        explicit TokenVectorMapper(int vector_dim = 128) noexcept
            : vector_dim_(vector_dim), rng_(std::random_device{}()) {}

        // Initialize embeddings from BPE instance
        void initializeFromBPE(const BytePairEncoding& bpe);

        // Initialize embeddings for tokens from set
        void initializeFromTokens(const std::set<std::string>& tokens);

        // Add a single token with random initialization
        void addToken(const std::string& token);

        // Add a token with specific vector
        void addToken(const std::string& token, const std::vector<float>& vector);

        // Get vector for a token
        std::vector<float> getVector(const std::string& token) const;

        // Get vector by reference (for modifications)
        std::vector<float>& getVectorRef(const std::string& token);

        // Check if token exists
        bool hasToken(const std::string& token) const noexcept;

        // Get all tokens
        std::vector<std::string> getTokens() const;

        // Get embedding dimension
        int getVectorDim() const noexcept { return vector_dim_; }

        // Get vocabulary size
        int getVocabSize() const noexcept { return token_vectors_.size(); }

        // Save embeddings to file (separate from BPE)
        bool saveEmbeddings(const std::string& filename) const;

        // Load embeddings from file
        bool loadEmbeddings(const std::string& filename);

        // Combined save: BPE + embeddings
        bool saveWithBPE(const BytePairEncoding& bpe, const std::string& bpe_filename,
            const std::string& embeddings_filename) const;

        // Combined load: BPE + embeddings
        bool loadWithBPE(BytePairEncoding& bpe, const std::string& bpe_filename,
            const std::string& embeddings_filename);

        // Update vector for existing token
        bool updateVector(const std::string& token, const std::vector<float>& new_vector);

        // Get similarity between two tokens (cosine similarity)
        float getSimilarity(const std::string& token1, const std::string& token2) const;

        // Find most similar tokens to a given token
        std::vector<std::pair<std::string, float>> findSimilar(const std::string& token, int top_k = 5) const;

        // Encode a word using BPE and get vectors
        std::vector<std::vector<float>> encodeToVectors(const BytePairEncoding& bpe, const std::string& word) const;

    private:
        int vector_dim_;
        std::unordered_map<std::string, std::vector<float>> token_vectors_;
        mutable std::mt19937 rng_;

        // Initialize a random vector
        std::vector<float> initRandomVector() const;

        // Normalize vector (for cosine similarity)
        static void normalizeVector(std::vector<float>& vec);

        // Calculate dot product
        static float dotProduct(const std::vector<float>& a, const std::vector<float>& b);

        // Calculate vector magnitude
        static float magnitude(const std::vector<float>& vec);
    };


}*/

