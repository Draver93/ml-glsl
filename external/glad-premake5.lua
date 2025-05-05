project "glad"
   kind "StaticLib"
   language "C"
   staticruntime "on"
   systemversion "latest"

   targetdir ("bin/" .. outputdir .. "/%{prj.name}")
   objdir ("bin-int/" .. outputdir .. "/%{prj.name}")

   files {
      "src/**.c"
   }

   includedirs {
      "include"
   }

   filter "system:windows"
      systemversion "latest"

   filter "configurations:Debug"
      runtime "Debug"
      symbols "on"

   filter "configurations:Release"
      runtime "Release"
      optimize "on"