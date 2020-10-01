package cmd

import (
	"log"
	"os"

	"github.com/davidkleiven/gogafit/gafit"
	"github.com/spf13/cobra"
)

// predCmd represents the pred command
var predCmd = &cobra.Command{
	Use:   "pred",
	Short: "Command for predicting from a GA model",
	Long: `Command for making predictions from a GA model. It works on CSV dat formatted
the same way as the training data. See gogafit fit -h for examples. It calculates the prediction
as well as the estimated prediction error.

gogafit pred -m fitted_model.json -d dataToPredict.csv

note that dataToPredict.csv can also be the training data, in which case the computed values
are the in-sample predictions and prediction errors.
	`,
	Run: func(cmd *cobra.Command, args []string) {
		modelFile, err := cmd.Flags().GetString("model")
		if err != nil {
			log.Fatalf("%s\n", err)
			return
		}

		model, err := gafit.ReadModel(modelFile)
		if err != nil {
			log.Fatalf("%s\n", err)
			return
		}

		// Check if the datafile used by the model exists
		if _, err := os.Stat(model.Datafile); os.IsNotExist(err) {
			log.Fatalf("Looking for data at %s but can't find it\n", model.Datafile)
			return
		} else if err != nil {
			log.Fatalf("Error when checking file %s\n", err)
			return
		}

		data, err := gafit.Read(model.Datafile, "")

		predDataFile, err := cmd.Flags().GetString("data")
		if err != nil {
			log.Fatalf("%s\n", err)
			return
		}

		predData, err := gafit.Read(predDataFile, "")
		if err != nil {
			log.Fatalf("%s\n", err)
			return
		}

		pred := gafit.GetPredictions(data, model, &predData)
		outfile := predDataFile[:len(predDataFile)-4] + "_predictions.csv"
		err = gafit.SavePredictions(outfile, pred)

		if err != nil {
			log.Fatalf("%s\n", err)
			return
		}
		log.Printf("Predictions for the data in %s is written to %s\n", predDataFile, outfile)
	},
}

func init() {
	rootCmd.AddCommand(predCmd)

	predCmd.Flags().StringP("model", "m", "", "JSON file holding the model")
	predCmd.Flags().StringP("data", "d", "", "CSV file with data to predict")
}
