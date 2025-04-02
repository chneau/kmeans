# kmeans

k-means clustering algorithm implementation written in Go using generics.

## Example

```go
package main

import (
	"fmt"
	"math/rand"

	"github.com/chneau/kmeans"
)

type Numbers int64

func (e Numbers) Coordinates() []float64 {
	return []float64{float64(e)}
}

func ExampleNumbers() {
	dataset := []Numbers{
		// first cluster
		1, 2, 3,
		// second cluster
		11, 12, 13,
		// third cluster
		21, 22, 23,
		// outlier
		100,
	}
	k := 4
	deltaThreshold := 0.01
	iterationThreshold := 100
	rng := rand.New(rand.NewSource(0))
	clusters, err := kmeans.Cluster(dataset, k, deltaThreshold, iterationThreshold, rng)
	if err != nil {
		panic(err)
	}

	// Output: (Unordered)
	// Cluster 0: Observations = [1 2 3]
	// Cluster 1: Observations = [11 12 13]
	// Cluster 2: Observations = [21 22 23]
	// Cluster 3: Observations = [100]
	for i, cluster := range clusters {
		fmt.Printf("Cluster %d: Observations = %v\n", i, cluster)
	}
}

type Coordinates [2]int

func (c Coordinates) Coordinates() []float64 {
	return []float64{float64(c[0]), float64(c[1])}
}

func ExampleCoordinates() {
	dataset := []Coordinates{
		// first cluster
		{1, 2}, {2, 3}, {3, 4},
		// second cluster
		{11, 12}, {12, 13}, {13, 14},
		// third cluster
		{21, 22}, {22, 23}, {23, 24},
		// outlier
		{100, 200},
	}
	k := 4
	deltaThreshold := 0.01
	iterationThreshold := 100
	rng := rand.New(rand.NewSource(0))
	clusters, err := kmeans.Cluster(dataset, k, deltaThreshold, iterationThreshold, rng)
	if err != nil {
		panic(err)
	}

	// Output: (Unordered)
	// Cluster 0: Observations = [{1 2} {2 3} {3 4}]
	// Cluster 1: Observations = [{11 12} {12 13} {13 14}]
	// Cluster 2: Observations = [{21 22} {22 23} {23 24}]
	// Cluster 3: Observations = [{100 200}]
	for i, cluster := range clusters {
		fmt.Printf("Cluster %d: Observations = %v\n", i, cluster)
	}
}

func main() {
	ExampleNumbers()
	ExampleCoordinates()
}
```
