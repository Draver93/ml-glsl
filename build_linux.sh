#!/bin/bash
git submodule init
git submodule update --init --recursive

sudo apt install -y \
    build-essential \
    libtbb-dev \
    libgl1-mesa-dev \
    libxxf86vm-dev \
    libx11-dev \
    libxcursor-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxi-dev
    
vendor/premake5/premake5a15 gmake2