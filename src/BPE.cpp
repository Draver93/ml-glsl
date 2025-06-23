#include "BPE.h"

#include <unordered_set>
#include <algorithm>
#include <future>
#include <functional>

namespace NNGL {

    BPE::BPE(size_t mergeLimit) : m_MergeLimit(mergeLimit) {}

    void BPE::processChunk(const char* chunk, size_t chunkSize) {
        std::vector<std::shared_ptr<Token>> tokens;
        tokens.reserve(chunkSize);

        size_t i = 0;
        while (i < chunkSize) {
            std::lock_guard<std::mutex> lock(m_TrieMutex);
            auto [token, len] = m_TokenTrie.match(chunk, chunkSize, i);
            if (!token) {
                token = std::make_shared<Token>(chunk[i]);
                len = 1;
            }
            tokens.emplace_back(token);
            i += len;
        }

        std::unordered_map<std::shared_ptr<Token>, std::vector<int>, TokenHasher, TokenEqual> pairs;

        size_t mergeCount = 0;
        while (mergeCount++ < m_MergeLimit) {
            std::shared_ptr<Token> bestPair = nullptr;
            size_t maxFreq = 0;

            for (size_t i = 0; i < tokens.size() - 1; i++) {
                auto pair = std::make_shared<Token>(tokens[i], tokens[i + 1]);
                pairs[pair].push_back(i);
                if (pairs[pair].size() > maxFreq) {
                    bestPair = pair;
                    maxFreq = pairs[pair].size();
                }
            }

            if (!bestPair || maxFreq <= 1) break;

            auto& positions = pairs[bestPair];
            for (auto it = positions.rbegin(); it != positions.rend(); ++it) {
                int pos = *it;
                if (pos + 1 >= tokens.size()) continue;
                tokens[pos] = bestPair;
                tokens.erase(tokens.begin() + pos + 1);
            }
            pairs.clear();
        }

        std::unordered_set<std::shared_ptr<Token>, TokenHasher, TokenEqual> uniqueTokens(tokens.begin(), tokens.end());
        {
            std::lock_guard<std::mutex> lock(m_TrieMutex);
            for (const auto& t : uniqueTokens) {
                bool isNewToken = m_TokenTrie.insert(t->getStr(), t);
                if (isNewToken) { m_VocabSize++; }
            }
        }
    }

    void BPE::trainFromFiles(const std::vector<std::string>& filePaths, bool append) {
        if (!append) {
            m_TokenTrie.clear();
            m_VocabSize = 0; // Reset vocab size when not appending
        }

        std::vector<std::future<void>> futures;
        for (const auto& path : filePaths) {
            std::ifstream file(path, std::ios::binary);
            if (!file) throw std::runtime_error("Cannot open: " + path);

            std::vector<char> buffer(1024);
            while (file.read(buffer.data(), buffer.size()) || file.gcount() > 0) {
                std::streamsize actualSize = file.gcount();
                std::vector<char> chunk(buffer.begin(), buffer.begin() + actualSize);

                futures.emplace_back(std::async(std::launch::async, [this, chunk = std::move(chunk)]() {
                    processChunk(chunk.data(), chunk.size());
                }));
            }
        }

        for (auto& fut : futures) fut.get();
    }

    std::vector<std::string> BPE::tokenizeInput(const char* input, size_t inputLen) {
        std::vector<std::string> result;
        size_t i = 0;

        while (i < inputLen) {
            auto [token, len] = m_TokenTrie.match(input, inputLen, i);
            if (!token) {
                token = std::make_shared<Token>(input[i]);
                len = 1;
            }
            result.emplace_back(token->getStr());
            i += len;
        }

        return result;
    }

    void BPE::save(const std::string& filepath) const {
        std::ofstream file(filepath, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot open file for writing: " + filepath);

        std::function<void(const TrieNode*, const std::string&)> saveNode = [&](const TrieNode* node, const std::string& prefix) {
            if (node->token) {
                size_t len = prefix.size();
                file.write(reinterpret_cast<const char*>(&len), sizeof(len));
                file.write(prefix.c_str(), len);
                // Save usage score
                file.write(reinterpret_cast<const char*>(&node->usageScore), sizeof(node->usageScore));
            }

            for (const auto& [ch, child] : node->children) {
                saveNode(child.get(), prefix + ch);
            }
        };

        saveNode(&m_TokenTrie.root, "");
    }

    void BPE::load(const std::string& filepath) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot open file for reading: " + filepath);

        m_TokenTrie.root = TrieNode{};

        m_VocabSize = 0;
        while (file) {
            size_t len;
            if (!file.read(reinterpret_cast<char*>(&len), sizeof(len))) break;

            std::string tokenStr(len, '\0');
            if (!file.read(tokenStr.data(), len)) break;

            size_t usageScore;
            if (!file.read(reinterpret_cast<char*>(&usageScore), sizeof(usageScore))) break;

            auto token = std::make_shared<Token>(tokenStr[0]);
            for (size_t i = 1; i < tokenStr.size(); ++i) {
                auto nextChar = std::make_shared<Token>(tokenStr[i]);
                token = std::make_shared<Token>(token, nextChar);
            }

            m_VocabSize++;
            m_TokenTrie.insert(tokenStr, token, usageScore);
        }
    }

}