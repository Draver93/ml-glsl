@echo off
git submodule init
git submodule update --recursive
vendor\premake5\premake5.exe vs2019