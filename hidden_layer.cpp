#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <cmath>

#include "hidden_layer.h"

using namespace std;

hidden_layer::hidden_layer(int numInputs, int numHiddenUnits):
  numInputs(numInputs),
  numHiddenUnits(numHiddenUnits)
{
  weightRange = 1.0 / sqrt(numInputs);
  // weightRange = 0.0;
  init();
}

hidden_layer::hidden_layer(int numInputs,
                           int numHiddenUnits, float weightRange):
  numInputs(numInputs),
  numHiddenUnits(numHiddenUnits),
  weightRange(weightRange)
{
  init();
}

void hidden_layer::init()
{
  numWeights = numInputs * numHiddenUnits;
  weights = new float[numWeights];

  for (int i = 0; i < numWeights; i++)
  {
    weights[i] = RandomNumber(-weightRange, weightRange);
  }

  biases = new float[numHiddenUnits];

  for (int i = 0; i < numHiddenUnits; i++)
  {
    biases[i] = 0.0;
  }
}

void hidden_layer::printWeights(int n)
{
  n = min(numHiddenUnits, n);
  for (int i = 0; i < n; i++)
  {
    for (int j = 0; j < numInputs; j++)
    {
      cout << weights[i * numInputs + j] << " ";
    }
    cout << endl;
  }
}

void hidden_layer::sigmoidTransform(float *x)
{
  for (int i = 0; i < numHiddenUnits; i++)
  {
    x[i] = 1 / (1 + exp(-1 * x[i]));
  }
}

void hidden_layer::encode(float *input, float *output)
{
  for (int i = 0; i < numHiddenUnits; i++)
  {
    float sum = 0.0;
    for (int j = 0; j < numInputs; j++)
    {
      sum += weights[i * numInputs + j] * input[j];
    }
    output[i] = tanhTransform(sum);
  }
}

float hidden_layer::squared_loss(float *output, float t1, float t2)
{
  float error = (output[0] - t1) * (output[0] - t1);
  error += (output[1] - t2) * (output[1] - t2);
  return 0.5 * error;
}

void hidden_layer::compute_delta_output(float *delta, float *o, float t1,
                                        float t2)
{
  delta[0] = tanhDerivative(o[0]) * (o[0] - t1);
  delta[1] = tanhDerivative(o[1]) * (o[1] - t2);
}


void hidden_layer::compute_delta_hidden(float *delta_curr_layer,
                                        float *delta_next_layer,
                                        float *output_curr_layer,
                                        hidden_layer *next_layer)
{
  //i is actually j
  //j is actually k
  float *output_layer_weights = next_layer->weights;
  int numHidUnits_nextLayer = next_layer->numHiddenUnits;

  for (int i = 0; i < numHiddenUnits; i++)
  {
    float sum = 0.0;

    for (int j = 0; j < numHidUnits_nextLayer; j++)
    {
      //we are iterating through ith column of next layer's weight matrix
      sum += delta_next_layer[j] * output_layer_weights[j * numHiddenUnits + i];
    }
    delta_curr_layer[i] =
      tanhDerivative(output_curr_layer[i]) * sum;
  }
}

void hidden_layer::updateWeights(float *delta_curr_layer,
                                 float *output_prev_layer, float learn_rate)
{
  for (int i = 0; i < numHiddenUnits; i++)
  {
    for (int j = 0; j < numInputs; j++)
    {
      weights[i * numInputs + j] -=
        learn_rate * output_prev_layer[j] * delta_curr_layer[i];
    }
    // biases[i] -= learn_rate * delta_curr_layer[i];
  }
}

hidden_layer::~hidden_layer()
{
  delete[] weights;
  delete[] biases;
}
