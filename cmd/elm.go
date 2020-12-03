package cmd

import (
	"fmt"
	"log"
	"math/rand"
	"time"

	"github.com/davidkleiven/gogafit/elm"
	"github.com/davidkleiven/gogafit/gafit"
	"github.com/spf13/cobra"
)

// elmCmd represents the elm command
var elmCmd = &cobra.Command{
	Use:   "elm",
	Short: "Create an extreme learning machine network",
	Long: `Calculates the output of a hiddan layer in an extreme learning machine.
The hidden layer consists of the given number of activation functions.

gogafit elm -d mydata.csv -r 100 -s 200 -t feat3 

creates an ELM with 100 relu neurons and 200 sigmoid neurons. Similar to the other commands,
the format of the data file is

feat1, feat2, feat3
0.2, 0.1, 0.4
...

where the name of the column corresponding to the target feature is specified via the -y flag.
	`,
	Run: func(cmd *cobra.Command, args []string) {
		dataFile, err := cmd.Flags().GetString("data")
		if err != nil {
			log.Fatalf("%s\n", err)

		}

		target, err := cmd.Flags().GetString("target")
		if err != nil {
			log.Fatalf("%s\n", err)
			return
		}
		target, err = ClosestHeaderName(dataFile, target)
		if err != nil {
			log.Fatalf("%s\n", err)
			return
		}
		log.Printf("Using %s as target column\n", target)

		numRelu, err := cmd.Flags().GetUint("relu")
		if err != nil {
			log.Fatalf("%s\n", err)
			return
		}

		numSig, err := cmd.Flags().GetUint("sig")
		if err != nil {
			log.Fatalf("%s\n", err)
			return
		}

		dataset, err := gafit.Read(dataFile, target)
		if err != nil {
			log.Fatalf("%s\n", err)
			return
		}
		nWeights := dataset.NumFeatures()

		neurons := []elm.Neuron{}
		names := []string{}

		src := rand.NewSource(time.Now().UnixNano())
		rng := rand.New(src)

		// Add relu neurons
		for i := 0; i < int(numRelu); i++ {
			neurons = append(neurons, elm.RandomReluNeuronFactory(nWeights, rng))
			names = append(names, fmt.Sprintf("relu%d", i))
		}

		// Add sigmoid neurons
		for i := 0; i < int(numSig); i++ {
			neurons = append(neurons, elm.RandomSigmoidNeuronFactory(nWeights, rng))
			names = append(names, fmt.Sprintf("sigmoid%d", i))
		}

		G := elm.HiddenLayerMatrix(dataset.X, neurons)
		outfile := dataFile[:len(dataFile)-4] + "_elm.csv"
		gafit.Write(outfile, G, dataset.Y, names, dataset.TargetName)
		log.Printf("Data for ELM written to %s\n", outfile)
	},
}

func init() {
	rootCmd.AddCommand(elmCmd)

	elmCmd.Flags().StringP("data", "d", "", "Datafile with inputs for the input layer")
	elmCmd.Flags().StringP("target", "y", "", "Name of columns that represent the target values")
	elmCmd.Flags().UintP("relu", "r", 1, "Number of rectifier activation functions in the hidden layer")
	elmCmd.Flags().UintP("sig", "s", 1, "Number of sigmoid activation functions in the hidden layer")
}
