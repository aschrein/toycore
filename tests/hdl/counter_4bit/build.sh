# cd in the current script
cd $(dirname $0)
# if build directory does not exist, create it
if [ ! -d build ]; then
    mkdir build
fi
# cd in the build directory
cd build
# print the current directory
echo "Current directory: $(pwd)"
# run the build command
yosys ../synthesize.ys