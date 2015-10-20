#include <iostream>
#include <boost/python.hpp>
#include <numpy.hpp>

#include "maze.h"
#include "neural_network.h"
// #include "simple_network.h"

using namespace std;
namespace bp = boost::python;
namespace np = boost::numpy;

neural_network *net = NULL;
bool verbose = false;
char maze_path[500] = "easy_maze.txt";


//execute a timestep of the maze simulation evaluation
double MazesimStep(Environment* _env, neural_network* _net)
{
	_env->generate_neural_inputs(_net->o_i);
	_net->Activate();

	//use the net's outputs to change heading and velocity of navigator
	_env->interpret_outputs(_net->o_k[0], _net->o_k[1]);
	//update the environment
	_env->Update();

	double dist = _env->distance_to_target();
	return dist;
}

int main(int argc, char *argv[])
{
	if (argc != 2)
	{
		cout << "usage: ./maze_exec [path_to_maze_file]" << endl;
		exit(1);
	}

	Environment *env = new Environment(argv[1]);
	neural_network *net = new neural_network(11, 8, 2, 0.0);

	const int timesteps = 400;
	double fitness = 0.0;

	for (int i = 0; i < timesteps; i++)
	{
		fitness += MazesimStep(env, net);
	}
	// env->display();
	cout << "final location: " << env->hero.location.x << " " <<
	     env->hero.location.y << endl;
	//calculate fitness of individual as closeness to target
	fitness = 300.0 - env->distance_to_target();
	if (fitness < 0.1) fitness = 0.1;

	delete env;
	delete net;

	cout << "fitness: " << fitness << endl;
	return fitness;
}

void SetUp()
{
	if (net)
		delete net;
	net = new neural_network(11, 8, 2, 0.0);
}

void SetVerbosity(const bool _verbose)
{
	verbose = _verbose;
}

bp::list EvalNetwork(np::ndarray &_genes,
                     double successThres = 295.0)
{
	if (!net)
		return bp::list();

	vector<pair<float, float> > locationHistory;
	locationHistory.reserve(400);
	net->SetWeights(_genes);
	Environment *env = new Environment(maze_path);

	const int timesteps = 400;
	double fitness = 0.0;

	for (int i = 0; i < timesteps; i++)
	{
		fitness += MazesimStep(env, net);
		if (verbose)
		{
			locationHistory.push_back(
			  make_pair(env->hero.location.x, env->hero.location.y));
		}
	}

	//calculate fitness of individual as closeness to target
	fitness = 300.0 - env->distance_to_target();
	if (fitness < 0.1) fitness = 0.1;

	bp::list list;
	list.append(fitness);
	list.append(env->hero.location.x);
	list.append(env->hero.location.y);
	if (verbose)
	{
		for (size_t i = 0; i < locationHistory.size(); i++)
		{
			list.append(bp::make_tuple(locationHistory[i].first,
			                           locationHistory[i].second));
		}
	}

	delete env;
	return list;
}

np::ndarray ReturnWeights()
{
	Py_intptr_t shape[1] = { net->GetNumWeights() };
	np::ndarray result = np::zeros(1, shape, np::dtype::get_builtin<float>());
	if (net)
	{
		net->GetWeights(result);
	}
	return result;
}

int GetNetworkWeightCount()
{
	if (!net)
		return -1;
	return net->GetNumWeights();
}

int GetNetworkHiddenUnitCount()
{
	if (!net)
		return -1;
	return net->GetNumHiddenUnits();
}

void SetMazePath(const char *_maze_path)
{
	strcpy(maze_path, _maze_path);
}

BOOST_PYTHON_MODULE(maze_lib)
{
	np::initialize();
	bp::def("SetUp", SetUp);
	bp::def("SetMazePath", SetMazePath);
	bp::def("SetVerbosity", SetVerbosity);
	bp::def("EvalNetwork", EvalNetwork);
	bp::def("ReturnWeights", ReturnWeights);
	bp::def("GetNetworkWeightCount", GetNetworkWeightCount);
	bp::def("GetNetworkHiddenUnitCount", GetNetworkHiddenUnitCount);
}
