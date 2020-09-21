package elm

import "math/rand"

// Neuron represents a neuron
type Neuron struct {
	Weights        []float64
	ActivationFunc ActivationFunc
}

// Activation calculates the actuvation of the current neuron
func (n Neuron) Activation(x []float64) float64 {
	return n.ActivationFunc(Dot(x, n.Weights))
}

// RandomNeuronFactory creates a neuron with randomly initialized weights
func RandomNeuronFactory(n int, rng *rand.Rand, a ActivationFunc) Neuron {
	neuron := Neuron{
		Weights:        make([]float64, n),
		ActivationFunc: a,
	}

	for i := range neuron.Weights {
		neuron.Weights[i] = rng.NormFloat64()
	}
	return neuron
}

// RandomSigmoidNeuronFactory is a convenience function that produce a neuron with random weights
// and a sigmoid activation
func RandomSigmoidNeuronFactory(n int, rng *rand.Rand) Neuron {
	return RandomNeuronFactory(n, rng, Sigmoid)
}

// RandomReluNeuronFactory is a convenience function that produce a neuron with random weights and
// a relu activation function
func RandomReluNeuronFactory(n int, rng *rand.Rand) Neuron {
	return RandomNeuronFactory(n, rng, Relu)
}
