package elm

import "gonum.org/v1/gonum/mat"

// Dot calculates the dot product between a and n
func Dot(a []float64, b []float64) float64 {
	if len(a) != len(b) {
		panic("Length of a and b must be the same")
	}

	value := 0.0
	for i := range a {
		value += a[i] * b[i]
	}
	return value
}

// EvaluateLayer evaluates the outut of each layer
func EvaluateLayer(neurons []Neuron, x []float64) []float64 {
	out := make([]float64, len(neurons))
	for i, neuron := range neurons {
		out[i] = neuron.Activation(x)
	}
	return out
}

func checkNeurons(numCols int, neurons []Neuron) bool {
	for _, neuron := range neurons {
		if len(neuron.Weights) != numCols {
			return false
		}
	}
	return true
}

// HiddenLayerMatrix calculates the matrix of a hidden layer given a matrix of input
// Each row of X corresponds to an input vector to the layer.
func HiddenLayerMatrix(X *mat.Dense, neurons []Neuron) *mat.Dense {
	r, c := X.Dims()

	if !checkNeurons(c, neurons) {
		panic("The number of weights in at least one neuron does not match the number of columns in X")
	}

	G := mat.NewDense(r, len(neurons), nil)

	for j := 0; j < r; j++ {
		res := EvaluateLayer(neurons, X.RawRowView(j))
		for i := range res {
			G.Set(j, i, res[i])
		}
	}
	return G
}
