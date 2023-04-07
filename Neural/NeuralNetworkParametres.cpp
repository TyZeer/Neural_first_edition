#include "NeuralNetworkParametres.h"

float NNParametres::SetNeuralNetworkMomentum() { return 0.6F; }
float NNParametres::SetNeuralNetworkLearnRate() { return 0.7F; }
bool NNParametres::DoAddBias() { return false; }
void NNParametres::SetNeuralNetworkConfiguration(std::vector<int>& NetConfig) {
	NetConfig.push_back(64);
	NetConfig.push_back(50);
	NetConfig.push_back(50);
	NetConfig.push_back(10);
}