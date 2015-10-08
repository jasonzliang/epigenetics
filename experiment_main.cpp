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
char maze_path[500] = "easy_maze.txt";


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

int main(int argc, char *argv[])
{
	if (argc != 2)
	{
		cout << "usage: ./maze_exec [path_to_maze_file]" << endl;
		exit(1);
	}

	Environment *env = new Environment(argv[1]);
	Network *net = new Network(11, 8, 2, true, false);

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

bp::list EvalNetwork(np::ndarray &_genes,
                     double successThres = 295.0)
{
	if (!net)
		return bp::list();

	vector<pair<float, float> > locationHistory;
	locationHistory.reserve(400);
	net->SetWeightAndActivity(_genes);
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
	// if (verbose)
	// 	env->display();

	//calculate fitness of individual as closeness to target
	fitness = 300.0 - env->distance_to_target();
	if (fitness < 0.1) fitness = 0.1;

	// update network weights using hebbian learning
	if (net->useAC)
	{
		// float learningRate = 0.0;
		// if (fitness > successThres)
		// 	learningRate = 0.05;
		float learningRate = pow(fitness / 300.0, 2);
		// we normalize net->hebbianAcitivty so largest abs value is 1
		float maxAbsValue = -1;
		for (int i = 0; i < net->numWeights; ++i)
		{
			maxAbsValue += fabs(net->hebbianActivity[i]);
			// if (fabs(net->hebbianActivity[i]) > maxAbsValue)
			// {
			// 	maxAbsValue = fabs(net->hebbianActivity[i]);
			// }
		}
		maxAbsValue /= net->numWeights;

		// cout << "learning rate: " << learningRate << endl;
		// cout << "maxAbsValue: " << maxAbsValue << endl;
		for (int i = 0; i < net->numWeights; ++i)
		{
			float update = learningRate *
			               (net->hebbianActivity[i] / maxAbsValue);
			// update = max(-0.25f * net->weightArray[i], min(0.25f * net->weightArray[i],
			//              update));
			net->weightArray[i] += update;
			net->weightArray[i] = max(-12.0f, min(12.0f, net->weightArray[i]));
			// cout << update << " ";
			// cout << net->weightArray[i] << " ";
		}
		// cout << endl;
	}

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
	Py_intptr_t shape[1] = { net->numWeights };
	np::ndarray result = np::zeros(1, shape, np::dtype::get_builtin<float>());
	if (net)
	{
		if (verbose)
		{
			for (int i = 0; i < net->numWeights; ++i)
			{
				cout << net->weightArray[i] << " ";
			}
			cout << endl;
		}
		copy(net->weightArray, net->weightArray + net->numWeights,
		     reinterpret_cast<float*>(result.get_data()));
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
	bp::def("ReturnWeights", ReturnWeights);
	bp::def("GetNetworkWeightCount", GetNetworkWeightCount);
	bp::def("GetNetworkHiddenUnitCount", GetNetworkHiddenUnitCount);
}
