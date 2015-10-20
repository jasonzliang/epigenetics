#ifndef NEURAL_NETWORK
#define NEURAL_NETWORK

#include <utility>
#include <vector>
#include <boost/python.hpp>
#include <numpy.hpp>

#include "hidden_layer.h"

class neural_network
{
public:
  hidden_layer *h;
  hidden_layer *o;
  float *o_i;
  float *o_j;
  float *o_k;
  float *delta_k;
  float *delta_j;
  float learn_rate;

  neural_network(int numInput, int numHidden, int numOutput, float learn_rate);

  void Train(std::vector<std::vector<float> > trainingInputs,
             std::vector<std::pair<float, float> > trainingTargets,
             int numOuterIter, int numTrainingExamples);

  float Backprop(std::vector<float> trainingInput, float t1, float t2);

  void Activate();

  void SetWeights(boost::numpy::ndarray &_weights);

  void GetWeights(boost::numpy::ndarray &_ndarray);

  int GetNumWeights()
  {
    return this->h->numWeights + this->o->numWeights;
  }

  int GetNumHiddenUnits()
  {
    return this->h->numHiddenUnits;
  }

  ~neural_network();
};

#endif
