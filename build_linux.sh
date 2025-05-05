#!/bin/bash
git submodule init
git submodule update --recursive
vendor/premake5/premake5 gmake