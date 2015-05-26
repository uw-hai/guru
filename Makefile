all: pomdpsol

pomdpsol:
	/opt/gcc-4.9.2/bin/g++ -O3 -std=c++11 ./pomdpsol.cpp -o bin/pomdpsol -I../ai-toolbox/include -L../ai-toolbox/build -lAIToolboxMDP -lAIToolboxPOMDP -lboost_program_options
