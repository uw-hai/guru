all:
	g++ -O3 -std=c++11 ./solve1.cpp -o work_learn -I../ai-toolbox/include -I./include -L../ai-toolbox/build -L./build -lAIToolboxMDP -lAIToolboxPOMDP -lWorkLearn -llpsolve55 -ldl -lcolamd -lboost_program_options -lboost_system -lboost_filesystem
