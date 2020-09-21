package elm

import "math"

// ActivationFunc function is a function that takes a value as input and returns a float
// value corresponding to the output of a neuron
type ActivationFunc func(x float64) float64

// Sigmoid is 1/(1+e^{-x})
func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// Relu is a rectifying function returnin max(0, x)
func Relu(x float64) float64 {
	if x < 0.0 {
		return 0.0
	}
	return x
}
