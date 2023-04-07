#pragma once
#include "INeuralNetworkParametres.h"
#include <vector>
#include "Neuron.h"
#include "Layer.h"
#include <iostream>

class NeuralNetwork
{
private:
	float Learn_rate;
	float Momentum;
	bool Do_add_bias;
	bool Isinitialized = false;
	std::vector<Layer> Layer_vect_;
	std::vector<int> Network_configuration;
	std::vector<float> OutputDifference;
public:
	NeuralNetwork(INeuralNetworkParametres& params);
	void InitSynapseInLayer(int layernum, std::function<float()> Initfunc);
	void SetInputNeuron(int position, float value);
	void Calculate(std::function<float(float)> act);
	void CalcOutputError(int position, float true_value, std::function<float(float)> act_deriv);
	void BackPropogationError(std::function<float(float)> act_deriv);
	float MediumSquareError();
	void ShowNeuralNetworkParametres(std::ostream& output);
	void ShowOutputNeuronValue(std::ostream& output, int output_layer);
};


 