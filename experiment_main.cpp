#include "maze.h"
#include "simple_network.h"

int main()
{
	Environment *enviornment = new Environment("medium_maze.txt");
	Network *network = new Network(5, 7, 2, false);
}
