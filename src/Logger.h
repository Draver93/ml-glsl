#pragma once

#include <iostream>
#include <string>

namespace NNGL {
    class Logger {
    public:
        static Logger& getInstance() {
            static Logger instance;
            return instance;
        }

        void log(const std::string& message) {
            if (m_enabled) {
                std::cout << message << std::endl;
            }
        }

        void setEnabled(bool enabled) {
            m_enabled = enabled;
        }

        bool isEnabled() const {
            return m_enabled;
        }

    private:
        Logger() : m_enabled(true) {}
        Logger(const Logger&) = delete;
        Logger& operator=(const Logger&) = delete;

        bool m_enabled;
    };

    // Convenience macro for logging
    #define LOG(message) NNGL::Logger::getInstance().log(message)
} 