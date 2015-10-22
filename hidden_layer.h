#ifndef HIDDEN_LAYER
#define HIDDEN_LAYER

#include <cstdlib>
#include <cmath>

class hidden_layer
{
public:
  int numInputs;
  int numHiddenUnits;
  int numWeights;
  float weightRange;

  float *weights;
  float *biases;

  hidden_layer(int numInputs, int numHiddenUnits);

  hidden_layer(int numInputs, int numHiddenUnits, float weightRange);

  virtual ~hidden_layer();

  void init();

  void printWeights(int n);

  void encode(float *input, float *output);

  void sigmoidTransform(float *x);

  float squared_loss(float *output, float t1, float t2);

  void compute_delta_output(float *delta, float *o, float t1,
                            float t2);

  void compute_delta_hidden(float *delta_curr_layer,
                            float *delta_next_layer,
                            float *output_curr_layer,
                            hidden_layer *next_layer);

  void updateWeights(float *delta_curr_layer,
                     float *output_prev_layer, float learn_rate);

  inline float RandomNumber(float Min, float Max)
  {
    return ((float(rand()) / float(RAND_MAX)) * (Max - Min)) + Min;
  }

  inline float sigmoidTransform(float x)
  {
    return 1.0 / (1.0 + exp(-1.0 * x));
  }

  inline float sigmoidDerivative(float x)
  {
    return x * (1.0 - x);
  }

  inline float tanhTransform(float x)
  {
    return tanh(x);
  }

  inline float tanhDerivative(float x)
  {
    return 1 - (x * x);
  }

  inline void setWeights(float *newWeights)
  {
    weights = newWeights;
  }

  inline void setEncodeBiases(float *newBiases)
  {
    biases = newBiases;
  }
};

#endif
