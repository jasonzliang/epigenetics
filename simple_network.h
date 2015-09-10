#ifndef SIMPLE_NETWORK_H
#define SIMPLE_NETWORK_H

#include <cmath>

class Network
{
public:
	const int numInput, numHidden, numOutput, numWeights;
	const bool useAC;
	float *weightArray, *inputUnits, *hiddenUnits, *outputUnits;
	int *activityCounter;
	bool *hiddenMask;

	Network(const int _i, const int _h, const int _o, const bool _useAC):
		numInput(i), numHidden(h), numOutput(o), useAC(_useAC)
	{
		this->numWeights = (i + o) * h;
		this->inputUnits = new float[i];
		this->hiddenUnits = new float[h];
		this->outputUnits = new float[o];
		this->activityCounter = new int[h];
		this->weightArray = new float[this->numWeights];

		for (int i = 0; i < this->numWeights; ++i)
		{
			this->weightArray = 0.0;
		}

		for (int i = 0; i < this->numHidden; ++i)
		{
			this->activityCounter = 0;
		}
	}

	~Network()
	{
		delete[] this->inputUnits;
		delete[] this->hiddenUnits;
		delete[] this->outputUnits;
		delete[] this->activityCounter;
		delete[] this->weightArray;
	}

	float inline Sigmoid(const float _input)
	{
		return 1.0 / (1.0 + exp(-_input));
	}

	float inline TanH(const float _input)
	{
		return tanh(_input);
	}

	void SetWeightAndActivityCounter(const float *_weights, const int *_activity,
	                                 const bool _reverse = false)
	{
		for (int i = 0; i < this->numWeights; ++i)
		{
			this->weightArray[i] = _weights[i];
		}
		for (int i = 0; i < this->numHidden; ++i)
		{
			this->activityCounter[i] = _activity[i];
		}
		this->SetHiddenMask(_reverse);
	}

	void SetHiddenMask(const bool _reverse)
	{
		float sum = 0.0;
		for (int i = 0; i < numHidden; ++i)
		{
			sum += static_cast<float>(this->activityCounter[i]);
		}
		for (int i = 0; i < numHidden; ++i)
		{
			const float prob = static_cast<float>(this->activityCounter[i]) / sum;
			const float r = static_cast <float> (rand()) /
			                static_cast <float> (RAND_MAX);

			if (r < prob)
			{
				this->hiddenMask[i] = _reverse;
			}
			else
			{
				this->hiddenMask[i] = !_reverse;
			}
		}
	}

	void UpdateActivityCounter()
	{
		int maxIndex = -1;
		float maxValue = -1;
		for (int i = 0; i < numHidden; ++i)
		{
			float temp = fabs(this->hiddenUnits[i]);
			if (temp > maxValue)
			{
				maxIndex = i;
				maxValue = temp;
			}
		}

		this->activityCounter[maxIndex]++;
	}

	void Activate()
	{
		float sum;

		for (int i = 0; i < numHidden; ++i)
		{
			if (_useAC && !this->hiddenMask[i])
			{
				this->hiddenUnits[i] = 0.0;
				continue;
			}
			sum = 0;
			for (int j = 0; j < numInput; ++j)
			{
				sum += this->weightArray[j + i * numInput] * this->inputUnits[j];
			}
			this->hiddenUnits[i] = this->TanH(sum);
		}

		// we find out which hidden unit is most active
		// (ie it has the largest output value)
		// and increment its counter
		this->UpdateActivityCounter();

		for (int i = 0; i < numOutput; ++i)
		{
			sum = 0;
			for (int j = 0; j < numHidden; ++j)
			{
				sum += this->weightArray[j + i * numHidden] * this->hiddenUnits[j];
			}
			this->outputUnits[i] = this->TanH(sum);
		}
	}
};

#endif