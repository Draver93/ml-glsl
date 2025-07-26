#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <fstream>
#include <memory>
#include <mutex>
#include <algorithm>
#include <functional>

namespace NNGL {

    struct Token {
        char c = '\0';
        std::size_t hash;
        std::shared_ptr<Token> lVal = nullptr, rVal = nullptr;

        Token() = delete;
        explicit Token(char val) : c(val), hash(std::hash<char>{}(val)) {}
        Token(char valA, std::shared_ptr<Token> tokB) : c(valA), rVal(tokB) {
            hash = std::hash<char>{}(c);
            combine_hash(hash, tokB->hash);
        }
        Token(std::shared_ptr<Token> tokA, std::shared_ptr<Token> tokB) : lVal(tokA), rVal(tokB) {
            hash = tokA->hash;
            combine_hash(hash, tokB->hash);
        }

        std::string getStr() const {
            std::string result;
            if (lVal) result += lVal->getStr();
            else result += c;
            if (rVal) result += rVal->getStr();
            return result;
        }

        bool operator==(const Token& other) const {
            return c == other.c &&
                bool(lVal) == bool(other.lVal) &&
                (!lVal || *lVal == *other.lVal) &&
                bool(rVal) == bool(other.rVal) &&
                (!rVal || *rVal == *other.rVal);
        }

        bool operator!=(const Token& other) const { return !(*this == other); }

    private:
        static void combine_hash(std::size_t& seed, std::size_t val) {
            seed ^= val + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
    };

    struct TokenHasher {
        std::size_t operator()(const std::shared_ptr<Token>& t) const noexcept { return t->hash; }
    };

    struct TokenEqual {
        bool operator()(const std::shared_ptr<Token>& a, const std::shared_ptr<Token>& b) const {
            if (a == b) return true;
            if (!a || !b) return false;
            return *a == *b;
        }
    };

    struct TrieNode {
        size_t usageScore = 0;
        std::shared_ptr<Token> token = nullptr;
        std::unordered_map<char, std::unique_ptr<TrieNode>> children;
    };

    class BPE;
    class TokenTrie {
    public:
        friend BPE;
    private:
        std::unordered_map<int, std::string> m_IdToToken;
        std::unordered_map<std::string, int> m_TokenToId;
    public:
        TrieNode root;

        bool insert(const std::string& sequence, std::shared_ptr<Token> token, size_t usageScore = 0) {
            TrieNode* node = &root;
            bool isNew = false;

            for (char c : sequence) {
                if (!node->children[c]) {
                    node->children[c] = std::make_unique<TrieNode>();
                    isNew = true;
                }
                node = node->children[c].get();
            }
            node->token = token;
            node->usageScore = usageScore;

            auto it = m_TokenToId.find(sequence);
            if (it == m_TokenToId.end()) {
                size_t id = m_TokenToId.size();
                m_TokenToId[sequence] = id;
                m_IdToToken[id] = sequence;
                return id;
            }

            return isNew;
        }

        std::pair<std::shared_ptr<Token>, size_t> match(const char* buffer, size_t length, size_t start) {
            TrieNode* node = &root;
            std::shared_ptr<Token> lastToken = nullptr;
            size_t matchLength = 0;

            for (size_t i = start; i < length; ++i) {
                auto it = node->children.find(buffer[i]);
                if (it == node->children.end()) break;

                node = it->second.get();
                if (node->token) {
                    lastToken = node->token;
                    matchLength = i - start + 1;
                }
            }

            if (lastToken) {
                node->usageScore++;
            }

            return { lastToken, matchLength };
        }

        void reduce(size_t maxTokens) {
            // First, collect all nodes with tokens that represent multi-character sequences
            std::vector<TrieNode*> tokenNodes;
            std::function<void(TrieNode*)> collect = [&](TrieNode* node) {
                if (node->token && (node->token->lVal || node->token->rVal)) {
                    // Only consider multi-character tokens for removal
                    tokenNodes.push_back(node);
                }
                for (auto& [_, child] : node->children) {
                    collect(child.get());
                }
            };
            collect(&root);

            // If we don't have too many tokens, no need to reduce
            if (tokenNodes.size() <= maxTokens) return;

            // Sort by usage score (ascending) to remove least used tokens first
            std::sort(tokenNodes.begin(), tokenNodes.end(),
                [](const TrieNode* a, const TrieNode* b) {
                    return a->usageScore < b->usageScore;
                });

            size_t toRemove = tokenNodes.size() - maxTokens;

            // Remove the least used tokens by clearing their token reference
            // We don't remove the nodes themselves as they might be part of paths to other tokens
            for (size_t i = 0; i < toRemove; ++i) {
                tokenNodes[i]->token = nullptr;
                tokenNodes[i]->usageScore = 0;
            }

            // Now clean up any orphaned nodes (nodes with no token and no children with tokens)
            std::function<bool(TrieNode*)> hasUsefulDescendants = [&](TrieNode* node) -> bool {
                if (node->token) return true;
                for (auto& [_, child] : node->children) {
                    if (hasUsefulDescendants(child.get())) return true;
                }
                return false;
            };

            std::function<void(TrieNode*)> pruneEmptyBranches = [&](TrieNode* node) {
                auto it = node->children.begin();
                while (it != node->children.end()) {
                    if (!hasUsefulDescendants(it->second.get())) {
                        it = node->children.erase(it);
                    }
                    else {
                        pruneEmptyBranches(it->second.get());
                        ++it;
                    }
                }
            };

            pruneEmptyBranches(&root);

            m_TokenToId.clear();
            m_IdToToken.clear();
            int nextId = 0;

            std::function<void(TrieNode*)> rebuildMaps = [&](TrieNode* node) {
                if (node->token) {
                    auto tokenStr = node->token->getStr();
                    if (m_TokenToId.find(tokenStr) == m_TokenToId.end()) {
                        m_TokenToId[tokenStr] = nextId;
                        m_IdToToken[nextId] = tokenStr;
                        ++nextId;
                    }
                }
                for (auto& [_, child] : node->children)
                    rebuildMaps(child.get());
            };
            rebuildMaps(&root);
        }

        const std::string& getTokenById(int id) {
            auto it = m_IdToToken.find(id);
            if (it == m_IdToToken.end()) 
                throw std::runtime_error("Token not found");
            return it->second;
        }

        int getIdByToken(const std::string& token) {
            auto it = m_TokenToId.find(token);
            if (it == m_TokenToId.end()) 
                throw std::runtime_error("Token not found");
            return it->second;
        }

        size_t getTokenCount() { return m_IdToToken.size(); }

        void clear() {
            root.children.clear();
            root.token = nullptr;
            m_IdToToken.clear();
            m_TokenToId.clear();
        }
    };

    class BPE {

    public:
        explicit BPE(size_t mergeLimit = 10000);

        void processChunk(const char* chunk, size_t chunkSize);
        void trainFromFiles(const std::vector<std::string>& files, bool append = true);
        void trainFromString(const std::string& text, bool append = true);
        void addToken(const std::string& token);
        std::vector<std::string> tokenizeInput(const char* input, size_t inputLen);
        void reduceVocab(size_t maxSize) { m_TokenTrie.reduce(maxSize); };
        size_t getVocabSize() { return m_TokenTrie.getTokenCount(); }
        const std::string& getTokenById(int id) { return m_TokenTrie.getTokenById(id); }
        size_t getTokenByName(const std::string &name) { return m_TokenTrie.getIdByToken(name); }

        void save(const std::string& filepath) const;
        void load(const std::string& filepath);

    private:
        std::mutex m_TrieMutex;
        TokenTrie m_TokenTrie;
        size_t m_MergeLimit;
    };

}