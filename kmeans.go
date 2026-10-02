package kmeans

import (
	"fmt"
	"math"
	"math/rand"
	"slices"
)

// Observation is an interface that represents a data point in n dimensions.
type Observation interface {
	Coordinates() []float64
}

// euclideanDistance calculates the Euclidean distance between two coordinate slices.
func euclideanDistance(a, b []float64) float64 {
	if len(a) != len(b) {
		panic("dimensions mismatch")
	}
	sum := 0.0
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return math.Sqrt(sum)
}

// Cluster implements the k-means clustering algorithm.
func Cluster[T Observation](dataset []T, k int, deltaThreshold float64, iterationThreshold int, rng *rand.Rand) ([][]T, error) {
	// Validate empty dataset
	if len(dataset) == 0 {
		return nil, fmt.Errorf("dataset is empty")
	}

	// Validate k
	if k <= 0 || k > len(dataset) {
		return nil, fmt.Errorf("invalid number of clusters: %d", k)
	}

	// Validate deltaThreshold
	if deltaThreshold <= 0 {
		return nil, fmt.Errorf("invalid delta threshold: %f", deltaThreshold)
	}

	// Validate iterationThreshold
	if iterationThreshold <= 0 {
		return nil, fmt.Errorf("invalid iteration threshold: %d", iterationThreshold)
	}

	// Validate rng
	if rng == nil {
		return nil, fmt.Errorf("random number generator is nil")
	}

	// Validate all observations have the same dimension
	dim := len(dataset[0].Coordinates())
	for _, obs := range dataset {
		if len(obs.Coordinates()) != dim {
			return nil, fmt.Errorf("inconsistent dimensions")
		}
	}

	// Handle the case where k is equal to the number of observations
	if k == len(dataset) {
		clusters := make([][]T, k)
		for i, obs := range dataset {
			clusters[i] = []T{obs}
		}
		return clusters, nil
	}

	// Handle the case where k is one
	if k == 1 {
		return [][]T{dataset}, nil
	}

	// Initialize centroids by randomly selecting k observations
	indices := make([]int, len(dataset))
	for i := range indices {
		indices[i] = i
	}
	rng.Shuffle(len(indices), func(i, j int) {
		indices[i], indices[j] = indices[j], indices[i]
	})
	centroids := make([][]float64, k)
	for j := range k {
		centroids[j] = slices.Clone(dataset[indices[j]].Coordinates())
	}

	// Assignment array to track which cluster each observation belongs to
	assignment := make([]int, len(dataset))

	// Main k-means loop
	for range iterationThreshold {
		// Assignment step: assign each observation to the nearest centroid
		for i := range dataset {
			minDist := math.Inf(1) // Positive infinity as initial distance
			minIndex := -1
			for j := range centroids {
				dist := euclideanDistance(dataset[i].Coordinates(), centroids[j])
				if dist < minDist {
					minDist = dist
					minIndex = j
				}
			}
			assignment[i] = minIndex
		}

		// Update step: calculate new centroids
		newCentroids := make([][]float64, k)
		for j := range newCentroids {
			newCentroids[j] = make([]float64, dim)
		}
		sums := make([][]float64, k)
		for j := range sums {
			sums[j] = make([]float64, dim)
		}
		counts := make([]int, k)

		// Compute sums and counts for each cluster
		for i, j := range assignment {
			coords := dataset[i].Coordinates()
			for d := range dim {
				sums[j][d] += coords[d]
			}
			counts[j]++
		}

		// Update centroids as the mean of assigned points
		for j := range k {
			if counts[j] > 0 {
				for d := range dim {
					newCentroids[j][d] = sums[j][d] / float64(counts[j])
				}
			}
		}

		// Reinitialize empty clusters. A cluster becomes empty when two
		// centroids are identical or end up closer to the same observations,
		// which is common when the initialization picks duplicate points.
		// Keeping the stale centroid in place would leave the cluster empty
		// forever, so move it onto the observation that is currently farthest
		// from the centroid it is assigned to. This pulls the empty centroid
		// towards an under-represented region of the dataset.
		used := make([]bool, len(dataset))
		for j := range k {
			if counts[j] > 0 {
				continue
			}
			bestIndex := -1
			bestDist := -1.0
			for i := range dataset {
				if used[i] {
					continue
				}
				dist := euclideanDistance(dataset[i].Coordinates(), centroids[assignment[i]])
				if dist > bestDist {
					bestDist = dist
					bestIndex = i
				}
			}
			if bestIndex == -1 {
				// No observation is available to seed the cluster, keep the
				// previous centroid rather than producing a zero centroid.
				newCentroids[j] = slices.Clone(centroids[j])
				continue
			}
			used[bestIndex] = true
			newCentroids[j] = slices.Clone(dataset[bestIndex].Coordinates())
		}

		// Check convergence by calculating the maximum centroid movement
		maxMovement := 0.0
		for j := range k {
			movement := euclideanDistance(centroids[j], newCentroids[j])
			if movement > maxMovement {
				maxMovement = movement
			}
		}

		// Update centroids for the next iteration
		centroids = newCentroids

		// Stop if maximum movement is below the threshold
		if maxMovement < deltaThreshold {
			break
		}
	}

	// Recompute the assignment against the final centroids so the returned
	// clusters reflect them. Without this, a cluster that was reinitialized on
	// the last iteration (for example when iterationThreshold is reached)
	// could still be returned as empty.
	for i := range dataset {
		minDist := math.Inf(1)
		minIndex := -1
		for j := range centroids {
			dist := euclideanDistance(dataset[i].Coordinates(), centroids[j])
			if dist < minDist {
				minDist = dist
				minIndex = j
			}
		}
		assignment[i] = minIndex
	}

	// Form clusters based on final assignments
	clusters := make([][]T, k)
	for i, obs := range dataset {
		j := assignment[i]
		clusters[j] = append(clusters[j], obs)
	}

	return clusters, nil
}
