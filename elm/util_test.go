package elm

import "gonum.org/v1/gonum/mat"
import "testing"

func TestHiddenLayerMatrix(t *testing.T) {
	neurons := []Neuron{
		Neuron{
			Weights: []float64{1.0, -1.0},
			ActivationFunc: Relu,
		},
		Neuron{
			Weights: []float64{0.0, -0.2},
			ActivationFunc: Sigmoid,
		},
	}

	X := mat.NewDense(2, 2, []float64{1.0, 2.0, 4.0, 3.0})
	want := mat.NewDense(2, 2, []float64{0.0, Sigmoid(-0.4), 1.0, Sigmoid(-0.6)})

	G := HiddenLayerMatrix(X, neurons)
	if !mat.EqualApprox(want, G, 1e-6) {
		t.Errorf("Expected\n%v\ngot\n%v\n", mat.Formatted(want), mat.Formatted(G))
	}
}
