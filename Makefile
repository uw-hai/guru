all:
	g++ -O3 -std=c++11 ./solve1.cpp -o work_learn -I../ai-toolbox/include -L../ai-toolbox/build -lAIToolboxMDP -lAIToolboxPOMDP -llpsolve55 -ldl -lcolamd -lboost_program_options
