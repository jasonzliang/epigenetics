#include <algorithm>
#include <utility>
#include <vector>
#include <boost/python.hpp>
#include <numpy.hpp>

#include "neural_network.h"

using namespace std;
namespace bp = boost::python;
namespace np = boost::numpy;

neural_network::neural_network(int numInput, int numHidden,
                               int numOutput, float learn_rate):
  learn_rate(learn_rate)
{
  // cout << "created a squared loss neural network with "
  // << numInput << " input, " << numHidden << " hidden, " << numOutput
  // << " output units and " << learn_rate << " learn rate" << endl;

  h = new hidden_layer(numInput, numHidden);
  o = new hidden_layer(numHidden, numOutput);
  // h->printWeights(9999);
  // o->printWeights(9999);

  o_i = new float[h->numInputs];
  o_j = new float[h->numHiddenUnits];
  o_k = new float[o->numHiddenUnits];

  delta_k = new float[o->numHiddenUnits];
  delta_j = new float[h->numHiddenUnits];
}

void neural_network::SetWeights(np::ndarray &_weights)
{
  char* arr_data = _weights.get_data();
  int stride = _weights.strides(0);
  for (int i = 0; i < _weights.shape(0); ++i)
  {
    float weight =
      static_cast<float>(*(reinterpret_cast<double*>(arr_data + i * stride)));
    if (i < h->numWeights)
    {
      h->weights[i] = weight;
    }
    else
    {
      o->weights[i - h->numWeights] = weight;
    }
  }
  // h->printWeights(9999);
  // o->printWeights(9999);
}

void neural_network::GetWeights(np::ndarray &_ndarray)
{
  copy(h->weights, h->weights + h->numWeights,
       reinterpret_cast<float*>(_ndarray.get_data()));

  copy(o->weights, o->weights + o->numWeights,
       reinterpret_cast<float*>(_ndarray.get_data()) + h->numWeights);

}

void neural_network::Train(vector<vector<float> > trainingInputs,
                           vector<pair<float, float> > trainingTargets,
                           int numOuterIter, int numTrainingExamples)
{
  for (int i = 0; i < numOuterIter; i++ )
  {
    float sum_squared_error = 0.0;
    for (int j = 0; j < numTrainingExamples; j++)
    {
      sum_squared_error += Backprop(trainingInputs[i],
                                    trainingTargets[j].first,
                                    trainingTargets[j].second);
    }
  }
}

float neural_network::Backprop(vector<float> trainingInput, float t1, float t2)
{
  for (size_t i = 0; i < trainingInput.size(); ++i)
  {
    o_i[i] = trainingInput[i];
  }

  h->encode(o_i, o_j);
  o->encode(o_j, o_k);

  o->compute_delta_output(delta_k, o_k, t1, t2);
  h->compute_delta_hidden(delta_j, delta_k, o_j, o);

  o->updateWeights(delta_k, o_j, learn_rate);
  h->updateWeights(delta_j, o_i, learn_rate);
  return o->squared_loss(o_k, t1, t2);
}

void neural_network::Activate()
{
  h->encode(o_i, o_j);
  o->encode(o_j, o_k);
}

neural_network::~neural_network()
{
  delete o;
  delete h;
  delete[] o_i;
  delete[] o_j;
  delete[] o_k;
  delete[] delta_k;
  delete[] delta_j;
}
