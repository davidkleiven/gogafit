package cmd

import (
	"encoding/csv"
	"log"
	"os"
	"strconv"

	"github.com/MaxHalford/eaopt"
	"github.com/davidkleiven/gogafit/gafit"
	"github.com/spf13/cobra"
	"gonum.org/v1/gonum/mat"
)

// fitCmd represents the fit command
var fitCmd = &cobra.Command{
	Use:   "fit",
	Short: "Fit data",
	Long: `Fit linear model from a datafile. Features are selected via a genetic algorithm
by minimizing the given cost function. The data should be organized in a csv file where the
first line is a header which assigns a name to each feature.

Example file
feat1, feat2, feat3
0.1, 0.5, -0.2
0.5, 0.1, 1.0

the program fits a coefficient vector c such that Xc = y. The y vector is called the target vector.
It is extracted from the target parameter passed (e.g. y is the column in the file whose name is
<target>). The remaining columns are taken as the X matrix.

Minimal example:

gogafit fit -d myfile.csv -t feat3

will take the columns corresponding to feat1 and feat2 as the X matrix, and use the last column
(here named feat3) as the y vector.
	`,
	Run: func(cmd *cobra.Command, args []string) {
		fitType, err := cmd.Flags().GetString("type")
		if err != nil {
			log.Fatalf("%s\n", err)
			return
		}

		dataFile, err := cmd.Flags().GetString("data")
		if err != nil {
			log.Fatalf("%s\n", err)
			return
		}

		mutRate, err := cmd.Flags().GetFloat64("mutrate")
		if err != nil {
			log.Fatalf("%s\n", err)
			return
		}

		ns, err := cmd.Flags().GetUint("csplits")
		if err != nil {
			log.Fatalf("%s\n", err)
			return
		}

		out, err := cmd.Flags().GetString("out")
		if err != nil {
			log.Fatalf("%s\n", err)
			return
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

		ng, err := cmd.Flags().GetUint("numgen")
		if err != nil {
			log.Fatalf("%s\n", err)
			return
		}

		cost, err := cmd.Flags().GetString("cost")
		if err != nil {
			log.Fatalf("%s\n", err)
			return
		}

		iprob, err := cmd.Flags().GetFloat64("iprob")
		if err != nil {
			log.Fatalf("%s\n", err)
			return
		}

		lograte, err := cmd.Flags().GetUint("lograte")
		if err != nil {
			log.Fatalf("%s\n", err)
			return
		}

		popsize, err := cmd.Flags().GetUint("popsize")
		if err != nil {
			log.Fatalf("%s\n", err)
			return
		}

		fdratio, err := cmd.Flags().GetFloat64("fdratio")
		if err != nil {
			log.Fatalf("%s\n", err)
			return
		}

		dataset, err := gafit.Read(dataFile, target)

		if err != nil {
			log.Fatalf("%s\n", err)
			return
		}

		if fitType != "reg" {
			log.Fatalf("Currently only regression is supported\n")
			return
		}

		// Initialize GA
		conf := eaopt.NewDefaultGAConfig()
		conf.PopSize = popsize
		ga, err := conf.NewGA()
		if err != nil {
			log.Fatalf("%s\n", err)
			return
		}

		// Set the number of generations to run for
		ga.NGenerations = ng

		callback := gafit.GABackupCB{
			Cost:       cost,
			Dataset:    dataset,
			DataFile:   dataFile,
			Rate:       lograte,
			BackupFile: out,
		}

		// Add a custom print function to track progress
		ga.Callback = callback.Build()

		// Initialize the linear model factory
		factory := gafit.LinearModelFactory{
			Config: gafit.LinearModelConfig{
				Data:               dataset,
				MutationRate:       mutRate,
				NumSplits:          ns,
				Cost:               getCostFunc(cost, dataset.NumFeatures()),
				MaxFeatToDataRatio: fdratio,
			},
			Prob: iprob,
		}

		// Find the minimum
		err = ga.Minimize(factory.Generate)
		if err != nil {
			log.Fatalf("%s\n", err)
			return
		}

		model := gafit.NewModel(ga.HallOfFame[0], dataset, cost, dataFile)
		gafit.SaveModel(out, model)
	},
}

func saveCoeff(fname string, features []string, coeff *mat.VecDense) {
	// Save features
	f, err := os.Create(fname)
	if err != nil {
		log.Fatalf("%s\n", err)
		return
	}
	defer f.Close()

	writer := csv.NewWriter(f)
	defer writer.Flush()

	for i := range features {
		valueString := strconv.FormatFloat(coeff.AtVec(i), 'f', 8, 64)
		record := []string{features[i], valueString}
		err = writer.Write(record)

		if err != nil {
			log.Fatalf("%s\n", err)
			return
		}
	}
}

func init() {
	rootCmd.AddCommand(fitCmd)

	// Here you will define your flags and configuration settings.

	// Cobra supports Persistent Flags which will work for this command
	// and all subcommands, e.g.:
	// fitCmd.PersistentFlags().String("foo", "", "A help for foo")

	// Cobra supports local flags which will only run when this command
	// is called directly, e.g.:
	fitCmd.Flags().StringP("type", "t", "reg", "Fit-type: regression (reg) or classify (cls)")
	fitCmd.Flags().StringP("data", "d", "", "Datafile. Should be stored in CSV format")
	fitCmd.Flags().StringP("target", "y", "lastCol", "Name of the column used as target in the fit")
	fitCmd.Flags().Float64P("mutrate", "m", 0.5, "Mutation rate in genetic algorithm")
	fitCmd.Flags().StringP("out", "o", "model.json", "File where the result of the best model is placed")
	fitCmd.Flags().UintP("numgen", "g", 100, "Number of generations to run")
	fitCmd.Flags().StringP("cost", "c", "aicc", "Cost function (aic|aicc|bic|ebic)")
	fitCmd.Flags().UintP("csplits", "s", 2, "Number of splits used for cross over operations")
	fitCmd.Flags().Float64P("iprob", "i", 0.5, "Probability of activating a feature in the initial pool of genomes")
	fitCmd.Flags().UintP("lograte", "r", 100, "Number generation between each log and backup of best solution")
	fitCmd.Flags().UintP("popsize", "p", 30, "Population size")
	fitCmd.Flags().Float64P("fdratio", "f", 0.8, "Maximum ratio between number of selected features and number of data points")
}

func getCostFunc(name string, numFeat int) gafit.CostFunction {
	switch name {
	case "aicc":
		return gafit.Aicc
	case "aic":
		return gafit.Aic
	case "bic":
		return gafit.Bic
	case "ebic":
		return gafit.NewDefaultEBic(numFeat).Evaluate
	default:
		if isScript(name) {
			hook := gafit.NewCostFunctionHook(name)
			return hook.Execute
		}
		log.Printf("Unknown cost function %s. Using default aicc instead\n", name)
		return gafit.Aicc
	}
}

func isScript(name string) bool {
	if _, err := os.Stat(name); os.IsNotExist(err) {
		return false
	}
	return true
}
