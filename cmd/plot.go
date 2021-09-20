package cmd

import (
	"log"
	"strings"

	"github.com/davidkleiven/gogafit/gafit"
	"github.com/spf13/cobra"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

// plotCmd represents the plot command
var plotCmd = &cobra.Command{
	Use:   "plot",
	Short: "Plot the fit in a scatter plot",
	Long: `Create a scatter plot of the predictions of one or multiple datasets.
If we have the training data in a file called train.csv and validation data in a file
validate.csv. Our trained model is stored in model.json, it can be plotted by

gogafit plot -d train.csv,validate.csv -m model.json -o plot.png
	`,
	Run: func(cmd *cobra.Command, args []string) {
		dataFiles, err := cmd.Flags().GetString("data")
		if err != nil {
			log.Fatalf("%s\n", err)
			return
		}

		modelFile, err := cmd.Flags().GetString("model")
		if err != nil {
			log.Fatalf("%s\n", err)
			return
		}

		out, err := cmd.Flags().GetString("out")
		if err != nil {
			log.Fatalf("%s\n", err)
			return
		}

		// Split dataFiles
		files := strings.Split(dataFiles, ",")
		if err != nil {
			log.Fatalf("%s\n", err)
			return
		}

		model, err := gafit.ReadModel(modelFile)
		if err != nil {
			log.Fatalf("Reading model: %s\n", err)
			return
		}

		// Initialize plot
		plt := plot.New()

		plt.X.Label.Text = model.TargetName + " predicted"
		plt.Y.Label.Text = model.TargetName + " reference"

		minValue := 1e100
		maxValue := -1e100
		colors := JosephAndHisBrothers()
		glyphs := NewDefaultGlyphCycle()

		for i, fname := range files {
			dataset, err := gafit.Read(fname, model.TargetName)
			if err != nil {
				log.Fatalf("Dataset %d: %s\n", i, err)
				return
			}
			pred := dataset.Dot(model.Coeffs)

			// Create points
			pts := make(plotter.XYs, pred.Len())
			for j := 0; j < pred.Len(); j++ {
				pts[j].Y = dataset.Y.AtVec(j)
				pts[j].X = pred.AtVec(j)

				if pts[j].Y > maxValue {
					maxValue = pts[j].Y
				}

				if pts[j].Y < minValue {
					minValue = pts[j].Y
				}
			}

			s, err := plotter.NewScatter(pts)
			s.GlyphStyle.Color = colors.Next()
			s.GlyphStyle.Shape = glyphs.Next()
			if err != nil {
				log.Fatalf("%s\n", err)
				return
			}
			plt.Add(s)
			plt.Legend.Add(fname)
		}

		rng := maxValue - minValue
		if rng < 1e-16 {
			rng = 1.0
		}

		straightLine := make(plotter.XYs, 2)
		frac := 0.05
		straightLine[0].X = minValue - frac*rng
		straightLine[0].Y = minValue - frac*rng
		straightLine[1].X = maxValue + frac*rng
		straightLine[1].Y = maxValue + frac*rng
		line, err := plotter.NewLine(straightLine)
		line.LineStyle.Color = colors.Get(2)
		if err != nil {
			log.Fatalf("%s\n", err)
			return
		}
		plt.Add(line)

		if err := plt.Save(4*vg.Inch, 4*vg.Inch, out); err != nil {
			log.Fatalf("Error while saving plot %s\n", err)
			return
		}
		log.Printf("Plot saved to %s\n", out)
	},
}

func init() {
	rootCmd.AddCommand(plotCmd)

	plotCmd.Flags().StringP("data", "d", "", "Comma separated list of datasets (e.g. test, train")
	plotCmd.Flags().StringP("model", "m", "", "JSON file with the model")
	plotCmd.Flags().StringP("out", "o", "gogafitPlot.png", "Image file where the model will be stored")
}
