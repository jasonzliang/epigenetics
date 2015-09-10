#include <iostream>
#include <boost/python.hpp>
#include <numpy.hpp>

#include "maze.h"
#include "simple_network.h"

using namespace std;
namespace bp = boost::python;
namespace np = boost::numpy;

Network *net = NULL;

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
	Environment *env = new Environment("medium_maze.txt");
	Network *net = new Network(11, 10, 2, false);

	const int timesteps = 400;
	double fitness = 0.0;

	for (int i = 0; i < timesteps; i++)
	{
		fitness += MazesimStep(env, net);
	}

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
	net = new Network(11, 10, 2, false);
}

void SetACFlag(const bool _value)
{
	if (net)
	{
		net->useAC = _value;
	}
}

float EvalNetwork(np::ndarray &_genes, np::ndarray &_ac)
{
	net->SetWeightAndActivityCounter(_genes, _ac);
	Environment *env = new Environment("medium_maze.txt");

	const int timesteps = 400;
	double fitness = 0.0;

	for (int i = 0; i < timesteps; i++)
	{
		fitness += MazesimStep(env, net);
	}

	//calculate fitness of individual as closeness to target
	fitness = 300.0 - env->distance_to_target();
	if (fitness < 0.1) fitness = 0.1;
	delete env;
	return fitness;
}

np::ndarray ReturnActivityCounter()
{
	Py_intptr_t shape[1] = { net->numHidden };
	np::ndarray result = np::zeros(1, shape, np::dtype::get_builtin<double>());
	if (net)
	{
		std::copy(net->activityCounter, net->activityCounter + net->numHidden,
		          reinterpret_cast<double*>(result.get_data()));
	}
	return result;
}

BOOST_PYTHON_MODULE(maze_lib)
{
	np::initialize();
	bp::def("SetUp", SetUp);
	bp::def("EvalNetwork", EvalNetwork);
	bp::def("SetACFlag", SetACFlag);
	bp::def("ReturnActivityCounter", ReturnActivityCounter);
}
