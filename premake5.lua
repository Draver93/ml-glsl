workspace "nn-glsl-core"
    configurations { "Debug", "Release" }
	architecture "x86_64"

outputdir = "%{cfg.buildcfg}-%{cfg.system}-%{cfg.architecture}"

dofile("external/glfw-premake5.lua")

project "nn-glsl-core"
    kind "ConsoleApp"
    language "C++"
    cppdialect "C++17"
    targetdir ("bin/" .. outputdir .. "/%{prj.name}")
    objdir ("bin-int/" .. outputdir .. "/%{prj.name}")
    staticruntime "On"
    dependson { "glfw" }

    files { "src/**.h", "src/**.cpp", "external/glad/src/glad.c" }
    includedirs {
        "external/glfw/include",
        "external/glad/include",
        "external/glm"
    }

    filter "configurations:Debug"
        defines { "DEBUG" }
        runtime "Debug"
        buildoptions { "/MTd" }
        symbols "On"

    filter "configurations:Release"
        defines { "NDEBUG" }
        runtime "Release"
        buildoptions { "/MT" }
        optimize "On"

    filter "system:windows"
        defines { "WINDOWS" }
        files { "external/glad/src/glad_wgl.c" }
        links {
            "glfw",
            "opengl32"
        }

    filter "system:linux"
        defines { "LINUX" }
        files { "external/glad/src/glad_glx.c" }
        links {
            "GL",         -- OpenGL
            "dl",
            "pthread",
            "X11",
            "Xrandr",
            "Xi",
            "Xxf86vm",
            "Xcursor",
            "glfw"        -- if installed via package manager
         }
   