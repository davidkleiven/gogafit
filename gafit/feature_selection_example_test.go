package gafit_test

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/MaxHalford/eaopt"
	"github.com/davidkleiven/gogafit/gafit"
	"gonum.org/v1/gonum/mat"
)

// Create a fictitious dataset
func sampleData() gafit.Dataset {
	data := gafit.Dataset{
		X:        mat.NewDense(20, 5, nil),
		Y:        mat.NewVecDense(20, nil),
		ColNames: []string{"const", "x", "x^2", "x^3", "x^4"},
	}

	for i := 0; i < 20; i++ {
		x := 0.1 * float64(i)
		for j := 0; j < 5; j++ {
			data.X.Set(i, j, math.Pow(x, float64(j)))
		}
		data.Y.SetVec(i, 5.0-2.0*x*x*x)
	}
	return data
}

func Example() {
	// Set a seed such that the run is deterministic
	rand.Seed(4)

	// Initialize GA with default configuration
	var ga, err = eaopt.NewDefaultGAConfig().NewGA()
	if err != nil {
		fmt.Println(err)
		return
	}

	// Set the number of generations to run for
	ga.NGenerations = 100

	// Add a custom print function to track progress
	ga.Callback = func(ga *eaopt.GA) {
		// Optionally print progress information (commented out for this example)
		// fmt.Printf("Best fitness at generation %d: %f\n", ga.Generations, ga.HallOfFame[0].Fitness)
	}

	// Initialize a dataset
	data := sampleData()

	// Initialize the linear model factory
	factory := gafit.LinearModelFactory{
		Config: gafit.LinearModelConfig{
			Data: data,

			// We use AICC as a measure of the quality of the model
			Cost: gafit.Aicc,
		},
	}

	// Find the minimum
	err = ga.Minimize(factory.Generate)
	if err != nil {
		fmt.Println(err)
		return
	}

	// Print the selected features
	best := ga.HallOfFame[0].Genome.(*gafit.LinearModel)

	// Run local optimization on the best genome
	res := best.Optimize()
	fmt.Printf("%v\n", data.IncludedFeatures(res.Include))

	// Output:
	// [const x^3]
}
