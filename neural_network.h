#ifndef NEURAL_NETWORK
#define NEURAL_NETWORK

#include <utility>
#include <vector>

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

  virtual void train(std::vector<std::vector<float> > trainingInputs,
                     std::vector<std::pair<float, float> > trainingTargets,
                     int numOuterIter, int numTrainingExamples);

  virtual float Backprop(std::vector<float> trainingInput, float t1, float t2);

  virtual void Activate();

  virtual ~neural_network();
};

#endif
