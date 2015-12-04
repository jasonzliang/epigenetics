#include <algorithm>
#include <iostream>
#include <utility>
#include <vector>
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
int timesteps = 400;
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

void SetUp(const float _learningRate, const int _timesteps)
{
	if (net)
		delete net;
	timesteps = _timesteps;
	net = new neural_network(11, 8, 2, _learningRate);
}

void SetVerbosity(const bool _verbose)
{
	verbose = _verbose;
}

bp::list EvalNetwork(np::ndarray &_genes)
{
	if (!net)
		return bp::list();

	vector<pair<float, float> > locationHistory;
	locationHistory.reserve(timesteps);
	net->SetWeights(_genes);
	Environment *env = new Environment(maze_path);

	double fitness = 0.0;
	int i;
	for (i = 0; i < timesteps; i++)
	{
		if (env->distance_to_target() < 5.0)
			break;
		fitness += MazesimStep(env, net);
		if (verbose)
		{
			locationHistory.push_back(
			  make_pair(env->hero.location.x, env->hero.location.y));
		}
	}

	//calculate fitness of individual as closeness to target
	fitness = (300.0 - env->distance_to_target()) + (400 - i);
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

bool sort_out_target(const out_target &a, const out_target &b)
{
	return a.diff > b.diff;
}

bp::tuple Backprop(np::ndarray &_master, np::ndarray &_student,
                   const int _epochs, const int _numsamples,
                   const bool _random)
{
	if (!net)
		return bp::tuple();

	net->SetWeights(_master);
	Environment *env = new Environment(maze_path);
	vector<vector<float> > inputs;
	inputs.reserve(timesteps);

	out_target prev_target;
	prev_target.o1 = 0.0;
	prev_target.o2 = 0.0;
	prev_target.diff = 0.0;
	vector<out_target> targets;
	targets.reserve(timesteps);

	for (int i = 0; i < timesteps; i++)
	{
		if (env->distance_to_target() < 5.0)
			break;
		env->generate_neural_inputs(net->o_i);
		vector<float> input;
		input.reserve(net->h->numInputs);
		for (int j = 0; j < net->h->numInputs; j++)
		{
			input.push_back(net->o_i[j]);
			// if (isnan(net->o_i[j]))
			// 	cout << "timestep: " << i << endl;
		}
		inputs.push_back(input);
		net->Activate();
		out_target target;
		target.o1 = net->o_k[0];
		target.o2 = net->o_k[1];
		target.diff = fabs(target.o1 - prev_target.o1) +
		              fabs(target.o2 - prev_target.o2);
		target.x = env->hero.location.x;
		target.y = env->hero.location.y;
		targets.push_back(target);
		env->interpret_outputs(net->o_k[0], net->o_k[1]);
		env->Update();
		prev_target = target;
	}

	if (_random)
	{
		random_shuffle(targets.begin(), targets.end());
	}
	else
	{
		sort(targets.begin(), targets.end(), sort_out_target);
		assert(targets[0].diff > targets[targets.size() - 1].diff);
	}
	targets.resize(_numsamples);

	net->SetWeights(_student);
	net->Train(inputs, targets, _epochs, _numsamples);


	bp::list list;
	for (size_t i = 0; i < targets.size(); ++i)
	{
		list.append(bp::make_tuple(targets[i].x, targets[i].y));
	}
	delete env;
	return bp::make_tuple(ReturnWeights(), list);
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

void PrintWeights()
{
	if (!net)
		return;
	net->h->printWeights(9999);
	net->o->printWeights(9999);
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
	bp::def("Backprop", Backprop);
	bp::def("PrintWeights", PrintWeights);
	bp::def("GetNetworkWeightCount", GetNetworkWeightCount);
	bp::def("GetNetworkHiddenUnitCount", GetNetworkHiddenUnitCount);
}
