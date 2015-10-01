#ifndef SIMPLE_NETWORK_H
#define SIMPLE_NETWORK_H

#include <cmath>
#include <ctime>
#include <cstdlib>
#include <iostream>
#include <boost/python.hpp>
#include <numpy.hpp>

void reseed(int val)
{
	unsigned short seed[3];

	seed[0] = val;
	seed[1] = val + 1;
	seed[2] = val + 2;
	seed48(seed);
}

class Network
{
public:
	int numInput, numHidden, numOutput;
	int numWeights;
	bool useAC, recurrent;
	float *weightArray, *inputUnits, *hiddenUnits, *outputUnits, *recurrentUnits;
	double *activityCounter, *tempActivityCounter;

	Network(const int _i, const int _h, const int _o, const bool _useAC = false,
	        const bool _recurrent = false):
		numInput(_i),
		numHidden(_h),
		numOutput(_o),
		useAC(_useAC),
		recurrent(_recurrent)
	{
		if (this->recurrent)
		{
			this->numInput += this->numHidden;
		}
		this->numWeights = (this->numInput + this->numOutput) * this->numHidden;

		this->inputUnits = new float[this->numInput];
		this->hiddenUnits = new float[this->numHidden];
		this->outputUnits = new float[this->numOutput];
		this->recurrentUnits = new float[this->numHidden];

		this->activityCounter = new double[this->numHidden];
		this->tempActivityCounter = new double[this->numHidden];
		this->weightArray = new float[this->numWeights];

		for (int i = 0; i < this->numWeights; ++i)
		{
			this->weightArray[i] = 0.0;
		}

		for (int i = 0; i < this->numHidden; ++i)
		{
			this->activityCounter[i] = 1.0;
			this->tempActivityCounter[i] = 0.0;
		}
	}

	~Network()
	{
		delete[] this->inputUnits;
		delete[] this->hiddenUnits;
		delete[] this->outputUnits;
		delete[] this->recurrentUnits;

		delete[] this->activityCounter;
		delete[] this->tempActivityCounter;
		delete[] this->weightArray;
	}

	float inline Sigmoid(const float _input) const
	{
		return 1.0 / (1.0 + exp(-_input));
	}

	float inline TanH(const float _input) const
	{
		return tanh(_input);
	}

	void SetWeightAndActivityCounter(boost::numpy::ndarray &_weights,
	                                 boost::numpy::ndarray &_activity,
	                                 const bool _reverse = false)
	{
		char* arr_data = _weights.get_data();
		int stride = _weights.strides(0);
		for (int i = 0; i < _weights.shape(0); ++i)
		{
			this->weightArray[i] =
			  static_cast<float>(*(reinterpret_cast<double*>(arr_data + i * stride)));
		}

		arr_data = _activity.get_data();
		stride = _activity.strides(0);
		double mean = 0.0;
		for (int i = 0; i < _activity.shape(0); ++i)
		{
			this->activityCounter[i] =
			  *(reinterpret_cast<double*>(arr_data + i * stride));
			mean += this->activityCounter[i];
			// we need to reset the temp activity counter
			this->tempActivityCounter[i] = 0.0;
		}
		mean /= static_cast<double>(this->numHidden);
		// we normalize activity counter by dividng by its mean
		for (int i = 0; i < _activity.shape(0); ++i)
		{
			this->activityCounter[i] /= mean;
		}
	}

	void UpdateTempActivityCounter()
	{
		int maxIndex = -1;
		float maxValue = -1.0;
		for (int i = 0; i < this->numHidden; ++i)
		{
			const float temp = fabs(this->hiddenUnits[i]);
			if (temp > maxValue)
			{
				maxIndex = i;
				maxValue = temp;
			}
		}
		this->tempActivityCounter[maxIndex] += 1.0;
	}

	void Activate()
	{
		float sum;

		for (int i = 0; i < numHidden; ++i)
		{
			sum = 0.0;
			for (int j = 0; j < this->numInput; ++j)
			{
				sum += this->weightArray[j + i * this->numInput] *
				       this->inputUnits[j];
			}
			this->hiddenUnits[i] = this->TanH(sum);
			if (this->useAC)
			{
				this->hiddenUnits[i] *= static_cast<float>(this->activityCounter[i]);
				// std::cout << this->hiddenUnits[i] << " ";
			}
		}

		if (this->recurrent)
		{
			for (int i = (this->numInput - this->numHidden); i < this->numInput; ++i)
			{
				this->inputUnits[i] = this->hiddenUnits[i];
			}
		}

		// we find out which hidden unit is most active
		// (ie it has the largest output value)
		// and increment its counter
		if (this->useAC)
		{
			this->UpdateTempActivityCounter();
		}

		const int offset = this->numHidden * this->numInput;
		for (int i = 0; i < this->numOutput; ++i)
		{
			sum = 0.0;
			for (int j = 0; j < this->numHidden; ++j)
			{
				sum += this->weightArray[j + (i * this->numHidden) + offset]
				       * this->hiddenUnits[j];
			}
			this->outputUnits[i] = this->TanH(sum);
		}
	}
};

#endif