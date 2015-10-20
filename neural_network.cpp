#include <utility>
#include <vector>

#include "neural_network.h"

using namespace std;

neural_network::neural_network(int numInput, int numHidden,
                               int numOutput, float learn_rate):
  learn_rate(learn_rate)
{
  // cout << "created a squared loss neural network with "
  // << numInput << " input, " << numHidden << " hidden, " << numOutput
  // << " output units and " << learn_rate << " learn rate" << endl;

  h = new hidden_layer(numInput, numHidden);
  o = new hidden_layer(numHidden, numOutput);

  o_i = new float[h->numInputs];
  o_j = new float[h->numHiddenUnits];
  o_k = new float[o->numHiddenUnits];

  delta_k = new float[o->numHiddenUnits];
  delta_j = new float[h->numHiddenUnits];
}

void neural_network::train(vector<vector<float> > trainingInputs,
                           vector<pair<float, float> > trainingTargets,
                           int numOuterIter, int numTrainingExamples)
{
  for (int i = 0; i < numOuterIter; i++ )
  {
    float sum_squared_error = 0.0;
    for (int j = 0; j < numTrainingExamples; j++)
    {
      sum_squared_error += backprop(trainingInputs[i],
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
