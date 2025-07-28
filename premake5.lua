workspace "ml-glsl"
    configurations { "Debug", "Release" }
    architecture "x86_64"
    startproject "ml-glsl-transformer"

outputdir = "%{cfg.buildcfg}-%{cfg.system}-%{cfg.architecture}"

dofile("external/glfw-premake5.lua")

project "ml-glsl-core"
    kind "StaticLib"
    language "C++"
    cppdialect "C++17"
    targetdir ("bin/" .. outputdir .. "/%{prj.name}")
    objdir ("bin-int/" .. outputdir .. "/%{prj.name}")
    staticruntime "On"
    dependson { "glfw" }

    -- Global defines for the entire project
    defines { "NOMINMAX" }

    files { "src/core/**.h", "src/core/**.cpp", "external/glad/src/glad.c" }
    includedirs {
        "external/glfw/include",
        "external/glad/include",
        "external/glm"
    }

    filter { "configurations:Debug" }
        defines { "DEBUG" }
        runtime "Debug"
        symbols "On"

    filter { "configurations:Release" }
        defines { "NDEBUG" }
        runtime "Release"
        optimize "On"

    filter { "configurations:Debug", "system:windows" }
        buildoptions { "/MTd" }
    filter { "configurations:Release", "system:windows" }
        buildoptions { "/MT" }

    filter { "configurations:Debug", "system:linux"}
        buildoptions { "-static-libgcc", "-static-libstdc++", "-g" }
    filter { "configurations:Release", "system:linux"}
        buildoptions { "-static-libgcc", "-static-libstdc++" }


    filter { "system:windows" }
        defines { "WINDOWS" }
        files { "external/glad/src/glad_wgl.c" }
        links {
            "glfw",
            "opengl32"
        }

    filter { "system:linux" }
        defines { "LINUX" }
        files { "external/glad/src/glad_glx.c" }
        links {
            "GL",
            "dl",
            "tbb",
            "pthread",
            "X11",
            "Xrandr",
            "Xi",
            "Xxf86vm",
            "Xcursor",
            "glfw"
        }

project "ml-glsl-tokenizer"
    kind "ConsoleApp"
    language "C++"
    cppdialect "C++17"
    targetdir ("bin/" .. outputdir .. "/%{prj.name}")
    objdir ("bin-int/" .. outputdir .. "/%{prj.name}")
    staticruntime "On"

    -- Global defines for the entire project
    defines { "NOMINMAX" }

    files { "src/tokenizer/**.h", "src/tokenizer/**.cpp", "external/glad/src/glad.c" }
    includedirs {
        "src/core",
        "external/glfw/include",
        "external/glad/include",
        "external/glm"
    }

    filter { "configurations:Debug" }
        defines { "DEBUG" }
        runtime "Debug"
        symbols "On"

    filter { "configurations:Release" }
        defines { "NDEBUG" }
        runtime "Release"
        optimize "On"

    filter { "configurations:Debug", "system:windows" }
        buildoptions { "/MTd" }
    filter { "configurations:Release", "system:windows" }
        buildoptions { "/MT" }

    filter { "configurations:Debug", "system:linux"}
        buildoptions { "-static-libgcc", "-static-libstdc++", "-g" }
    filter { "configurations:Release", "system:linux"}
        buildoptions { "-static-libgcc", "-static-libstdc++" }


    filter { "system:windows" }
        defines { "WINDOWS" }
        links {
            "ml-glsl-core"
        }

    filter { "system:linux" }
        defines { "LINUX" }
        links {
            "ml-glsl-core"
        }

project "ml-glsl-transformer"
    kind "ConsoleApp"
    language "C++"
    cppdialect "C++17"
    targetdir ("bin/" .. outputdir .. "/%{prj.name}")
    objdir ("bin-int/" .. outputdir .. "/%{prj.name}")
    staticruntime "On"
    dependson { "glfw" }

    -- Global defines for the entire project
    defines { "NOMINMAX" }

    files { "src/transformer/**.h", "src/transformer/**.cpp", "external/glad/src/glad.c" }
    includedirs {
        "src/core",
        "external/glfw/include",
        "external/glad/include",
        "external/glm"
    }

    filter { "configurations:Debug" }
        defines { "DEBUG" }
        runtime "Debug"
        symbols "On"

    filter { "configurations:Release" }
        defines { "NDEBUG" }
        runtime "Release"
        optimize "On"

    filter { "configurations:Debug", "system:windows" }
        buildoptions { "/MTd" }
    filter { "configurations:Release", "system:windows" }
        buildoptions { "/MT" }

    filter { "configurations:Debug", "system:linux"}
        buildoptions { "-static-libgcc", "-static-libstdc++", "-g" }
    filter { "configurations:Release", "system:linux"}
        buildoptions { "-static-libgcc", "-static-libstdc++" }


    filter { "system:windows" }
        defines { "WINDOWS" }
        files { "external/glad/src/glad_wgl.c" }
        links {
            "ml-glsl-core"
        }

    filter { "system:linux" }
        defines { "LINUX" }
        files { "external/glad/src/glad_glx.c" }
        links {
            "ml-glsl-core",
            "ml-glsl-tokenizer"
        }
