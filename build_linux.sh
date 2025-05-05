#!/bin/bash
git submodule init
git submodule update --init --recursive
vendor/premake5/premake5 gmake