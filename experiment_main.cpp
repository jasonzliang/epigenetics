#include <algorithm>
#include <iostream>
#include <boost/python.hpp>
#include <numpy.hpp>

#include "maze.h"
#include "simple_network.h"

using namespace std;
namespace bp = boost::python;
namespace np = boost::numpy;

Network *net = NULL;
bool verbose = false;
char maze_path[500]
  = "/home/jason/Desktop/Dropbox/fall2015/epigenetics/easy_maze.txt";


//execute a timestep of the maze simulation evaluation
double MazesimStep(Environment* _env, Network * _net)
{
	_env->generate_neural_inputs(_net->inputUnits);
	_net->Activate();

	//use the net's outputs to change heading and velocity of navigator
	_env->interpret_outputs(_net->outputUnits[0], _net->outputUnits[1]);
	//update the environment
	_env->Update();

	double dist = _env->distance_to_target();
	return dist;
}

int main()
{
	Environment *env = new Environment(
	  "/home/jason/Desktop/Dropbox/fall2015/epigenetics/easy_maze.txt");
	Network *net = new Network(11, 8, 2, false, false);

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

void SetUp(const bool _acflag, const bool _recurrent)
{
	if (net)
		delete net;
	net = new Network(11, 8, 2, _acflag, _recurrent);
}

void SetVerbosity(const bool _verbose)
{
	verbose = _verbose;
}

bp::tuple EvalNetwork(np::ndarray &_genes, np::ndarray &_ac,
                  double successThres = 295.0)
{
	if (!net)
		return bp::make_tuple();
	net->SetWeightAndActivityCounter(_genes, _ac);
	Environment *env = new Environment(maze_path);

	const int timesteps = 400;
	double fitness = 0.0;

	for (int i = 0; i < timesteps; i++)
	{
		fitness += MazesimStep(env, net);
	}
	if (verbose)
		env->display();
	//calculate fitness of individual as closeness to target
	fitness = 300.0 - env->distance_to_target();
	if (fitness < 0.1) fitness = 0.1;

	// if trial is successful, activity counter gets larger, otherwise gets
	// smaller
	if (net->useAC)
	{
		double base = 0.999;
		if (env->hero.location.y > 70.0 && fitness > successThres)
			base = 1.001;
		for (int i = 0; i < net->numHidden; ++i)
		{
			net->activityCounter[i] *= pow(base, net->tempActivityCounter[i]);
			net->activityCounter[i] =
			  max(0.1, min(10.0, net->activityCounter[i]));
		}
	}

	delete env;
	return bp::make_tuple(fitness, env->hero.location.x, env->hero.location.y);
}

np::ndarray ReturnActivityCounter()
{
	Py_intptr_t shape[1] = { net->numHidden };
	np::ndarray result = np::zeros(1, shape, np::dtype::get_builtin<double>());
	if (net)
	{
		copy(net->activityCounter, net->activityCounter + net->numHidden,
		     reinterpret_cast<double*>(result.get_data()));
	}
	return result;
}

int GetNetworkWeightCount()
{
	if (!net)
		return -1;
	return net->numWeights;
}

int GetNetworkHiddenUnitCount()
{
	if (!net)
		return -1;
	return net->numHidden;
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
	bp::def("ReturnActivityCounter", ReturnActivityCounter);
	bp::def("GetNetworkWeightCount", GetNetworkWeightCount);
	bp::def("GetNetworkHiddenUnitCount", GetNetworkHiddenUnitCount);
}
