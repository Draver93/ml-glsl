project "glfw"
   kind "StaticLib"
   language "C"
   staticruntime "on"
   systemversion "latest"

   targetdir ("bin/" .. outputdir .. "/%{prj.name}")
   objdir ("bin-int/" .. outputdir .. "/%{prj.name}")

   includedirs { "glfw/include" }

   files {
      "glfw/include/GLFW/*.h",
      "glfw/src/**.h",
      "glfw/src/**.c"
   }

   filter "system:windows"
      defines { "_GLFW_WIN32", "_CRT_SECURE_NO_WARNINGS" }

   filter "system:linux"
      defines { "_GLFW_X11" }
      pic "On"

   filter "configurations:Debug"
      runtime "Debug"
      symbols "On"

   filter "configurations:Release"
      runtime "Release"
      optimize "On"