package cmd

import (
	"log"

	"github.com/davidkleiven/gogafit/gafit"
	"github.com/spf13/cobra"
)

// polyCmd represents the poly command
var polyCmd = &cobra.Command{
	Use:   "poly",
	Short: "Add polynomial versions of a subset of the columns",
	Long: `Adds columns representing a polynomial version of a subset of the columns.
	
Example:
We have the following csv file named data.csv

feat1,feat2,targetQuantity
1.0,2.0,3.0
2.0,1.0,2.0

and we wich to add a new feature which is feat1^2. Run

gogafit poly -d data.csv -t targetQuantity -o 2 -p feat1

this will create a file data_poly.csv

feat1,feat2,feat1p2targetQuantity
1.0,2.0,1.0,3.0
2.0,1.0,4.0,2.0

	`,
	Run: func(cmd *cobra.Command, args []string) {
		dataFile, err := cmd.Flags().GetString("data")
		if err != nil {
			log.Fatalf("%s\n", err)
			return
		}

		pattern, err := cmd.Flags().GetString("pattern")
		if err != nil {
			log.Fatalf("%s\n", err)
			return
		}

		order, err := cmd.Flags().GetUint("order")
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
		log.Printf("Using %s as target value\n", target)

		data, err := gafit.Read(dataFile, target)
		cols := data.Columns(pattern)

		log.Printf("Slected columns:\n")
		for _, c := range cols {
			log.Printf("No. %d name: %s\n", c, data.ColNames[c])
		}

		newData := gafit.AddPoly(cols, data, int(order))

		// Store the result
		outfname := dataFile[:len(dataFile)-4] + "_poly.csv"
		if err = gafit.Write(outfname, newData.X, newData.Y, newData.ColNames, newData.TargetName); err != nil {
			log.Fatalf("%s\n", err)
			return
		}

		log.Printf("New dataset written to %s\n", outfname)
	},
}

func init() {
	rootCmd.AddCommand(polyCmd)

	polyCmd.Flags().StringP("data", "d", "", "Original datafile")
	polyCmd.Flags().StringP("pattern", "p", "", "Polynomial versions of all features containing this substring will be added")
	polyCmd.Flags().UintP("order", "o", 1, "Polynomial order")
	polyCmd.Flags().StringP("target", "y", "", "Name of the quantity used as target property")
}
