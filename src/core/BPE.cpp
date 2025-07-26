#include "BPE.h"

#include <unordered_set>
#include <algorithm>
#include <future>
#include <functional>

namespace NNGL {

    void clean_word(std::vector<char>& word) {
        word.erase(std::remove_if(word.begin(), word.end(),
            [](unsigned char c) { return std::ispunct(c) || c == '\n' || c == '\r'; }), word.end());
        std::transform(word.begin(), word.end(), word.begin(),
            [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    }

    BPE::BPE(size_t mergeLimit) : m_MergeLimit(mergeLimit) {}

    const char* BPE::save() {
        // Calculate total buffer size needed
        size_t header_size = sizeof(size_t) + sizeof(size_t); // merge limit + tokens data size

        // Allocate buffer (caller is responsible for freeing this memory)
        char* buffer = new char[getSaveSize()];
        char* ptr = buffer;

        // Write header: merge limit
        std::memcpy(ptr, &m_MergeLimit, sizeof(size_t));
        ptr += sizeof(size_t);
        

        for (int i = 0; i < m_TokenTrie.getTokenCount(); i++) {
            std::string token = m_TokenTrie.getTokenById(i);
            size_t len = token.size();
            std::memcpy(ptr, &len, sizeof(size_t));
            ptr += sizeof(size_t);
            std::memcpy(ptr, token.c_str(), len);
            ptr += len;
        }

        return buffer;
    }

    size_t BPE::getSaveSize() {
        size_t totalSize = sizeof(size_t); // merge limit

        for (int i = 0; i < m_TokenTrie.getTokenCount(); i++) {
            std::string token = m_TokenTrie.getTokenById(i);
            totalSize += sizeof(size_t);        // length field
            totalSize += token.size();         // string data
        }

        return totalSize;
    }

    BPE::BPE(const char* data, size_t dataSize) {
        m_TokenTrie.root = TrieNode{};
        const char* ptr = data;

        // Read header: merge limit
        if (ptr + sizeof(size_t) > data + dataSize)
            throw std::runtime_error("BPE save corrupted");

        std::memcpy(&m_MergeLimit, ptr, sizeof(size_t));
        ptr += sizeof(m_MergeLimit);


        const char* tokens_end = data + dataSize;

        // Read token data
        while (ptr < tokens_end) {
            // Read token length
            if (ptr + sizeof(size_t) > tokens_end)
                throw std::runtime_error("BPE save corrupted");

            size_t len;
            std::memcpy(&len, ptr, sizeof(size_t));
            ptr += sizeof(size_t);

            // Read token string data
            if (ptr + len > tokens_end)
                throw std::runtime_error("BPE save corrupted");

            std::string tokenStr(ptr, len);
            ptr += len;

            addToken(tokenStr);
        }
    }

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
            for (const auto& t : uniqueTokens) 
                m_TokenTrie.insert(t->getStr(), t);
        }
    }

    void BPE::trainFromFiles(const std::vector<std::string>& filePaths, bool append) {
        if (!append) {
            m_TokenTrie.clear();
        }

        std::vector<std::future<void>> futures;
        for (const auto& path : filePaths) {
            std::ifstream file(path, std::ios::binary);
            if (!file) throw std::runtime_error("Cannot open: " + path);

            std::vector<char> buffer(1024);
            while (file.read(buffer.data(), buffer.size()) || file.gcount() > 0) {
                std::streamsize actualSize = file.gcount();
                std::vector<char> chunk(buffer.begin(), buffer.begin() + actualSize);
                clean_word(chunk);

                futures.emplace_back(std::async(std::launch::async, [this, chunk = std::move(chunk)]() {
                    processChunk(chunk.data(), chunk.size());
                }));
            }
        }

        for (auto& fut : futures) fut.get();
    }

    void BPE::trainFromString(const std::string& text, bool append) {
        if (!append) {
            m_TokenTrie.clear();
        }

        // Process the text directly
        processChunk(text.c_str(), text.size());
    }

    void BPE::addToken(const std::string& token) {
        if (token.empty()) return;
        
        // Create a token object from the string
        auto tokenObj = std::make_shared<Token>(token[0]);
        for (size_t i = 1; i < token.size(); ++i) {
            auto nextChar = std::make_shared<Token>(token[i]);
            tokenObj = std::make_shared<Token>(tokenObj, nextChar);
        }
        
        // Insert directly into trie
        std::lock_guard<std::mutex> lock(m_TrieMutex);
        m_TokenTrie.insert(token, tokenObj);
    }

    std::vector<std::string> BPE::tokenizeInput(const char* input, size_t inputLen) {
        std::vector<std::string> result;
        size_t i = 0;

        while (i < inputLen) {
            auto [token, len] = m_TokenTrie.match(input, inputLen, i);
            if (!token) {
                //throw std::runtime_error("Critical! Token not found!");
                token = std::make_shared<Token>(input[i]);
                len = 1;
            }
            result.emplace_back(token->getStr());
            i += len;
        }

        return result;
    }

    void BPE::save(const std::string& filepath) {
        std::ofstream file(filepath, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot open file for writing: " + filepath);
        const char* data = save();
        size_t dataSize = getSaveSize();

        file.write(data, dataSize);
        delete[] data; // Clean up allocated memory
    }

    void BPE::load(const std::string& filepath) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot open file for reading: " + filepath);
        size_t data_size;
        {
            const auto begin = file.tellg();
            file.seekg(0, std::ios::end);
            const auto end = file.tellg();
            data_size = (end - begin);
            file.seekg(0, std::ios::beg);
        }

        m_TokenTrie.root = TrieNode{};
        size_t bytes_read = 0;
        // Read header: merge limit
        if (!file.read(reinterpret_cast<char*>(&m_MergeLimit), sizeof(m_MergeLimit))) {
            throw std::runtime_error("Failed to read merge limit from file");
        }
        bytes_read += sizeof(m_MergeLimit);

        // Read token data
        while (bytes_read < data_size && file) {
            size_t len;
            if (!file.read(reinterpret_cast<char*>(&len), sizeof(len))) break;
            bytes_read += sizeof(len);

            std::string tokenStr(len, '\0');
            if (!file.read(tokenStr.data(), len)) 
                break;
            bytes_read += len;

            auto token = std::make_shared<Token>(tokenStr[0]);
            for (size_t i = 1; i < tokenStr.size(); ++i) {
                auto nextChar = std::make_shared<Token>(tokenStr[i]);
                token = std::make_shared<Token>(token, nextChar);
            }

            addToken(tokenStr);
        }
    }

}