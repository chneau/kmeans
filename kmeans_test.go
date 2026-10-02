package kmeans

import (
	"math/rand"
	"slices"
	"testing"
)

type Numbers int64

func (e Numbers) Coordinates() []float64 {
	return []float64{float64(e)}
}

type Coordinates [2]int

func (c Coordinates) Coordinates() []float64 {
	return []float64{float64(c[0]), float64(c[1])}
}

func TestClusterNumbers(t *testing.T) {
	dataset := []Numbers{
		1, 2, 3,
		11, 12, 13,
		21, 22, 23,
		100,
	}
	k := 4
	deltaThreshold := 0.01
	iterationThreshold := 100
	rng := rand.New(rand.NewSource(0))

	clusters, err := Cluster(dataset, k, deltaThreshold, iterationThreshold, rng)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	expectedClusters := [][]Numbers{
		{1, 2, 3},
		{11, 12, 13},
		{21, 22, 23},
		{100},
	}

	// Check if the clusters match the expected clusters (unordered)
	if len(clusters) != len(expectedClusters) {
		t.Fatalf("expected %d clusters, got %d", len(expectedClusters), len(clusters))
	}

	matched := make([]bool, len(expectedClusters))
	for _, cluster := range clusters {
		found := false
		for i, expected := range expectedClusters {
			if !matched[i] && slices.Equal(cluster, expected) {
				matched[i] = true
				found = true
				break
			}
		}
		if !found {
			t.Errorf("unexpected cluster: %v", cluster)
		}
	}

	for i, matched := range matched {
		if !matched {
			t.Errorf("expected cluster %v not found", expectedClusters[i])
		}
	}
}

func TestClusterCoordinates(t *testing.T) {
	dataset := []Coordinates{
		{1, 2}, {2, 3}, {3, 4},
		{11, 12}, {12, 13}, {13, 14},
		{21, 22}, {22, 23}, {23, 24},
		{100, 200},
	}
	k := 4
	deltaThreshold := 0.01
	iterationThreshold := 100
	rng := rand.New(rand.NewSource(0))

	clusters, err := Cluster(dataset, k, deltaThreshold, iterationThreshold, rng)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	expectedClusters := [][]Coordinates{
		{{1, 2}, {2, 3}, {3, 4}},
		{{11, 12}, {12, 13}, {13, 14}},
		{{21, 22}, {22, 23}, {23, 24}},
		{{100, 200}},
	}

	// Check if the clusters match the expected clusters (unordered)
	if len(clusters) != len(expectedClusters) {
		t.Fatalf("expected %d clusters, got %d", len(expectedClusters), len(clusters))
	}

	matched := make([]bool, len(expectedClusters))
	for _, cluster := range clusters {
		found := false
		for i, expected := range expectedClusters {
			if !matched[i] && slices.Equal(cluster, expected) {
				matched[i] = true
				found = true
				break
			}
		}
		if !found {
			t.Errorf("unexpected cluster: %v", cluster)
		}
	}

	for i, matched := range matched {
		if !matched {
			t.Errorf("expected cluster %v not found", expectedClusters[i])
		}
	}
}

func TestClusterReinitializesEmptyClusters(t *testing.T) {
	// Centroids are sampled from the dataset, so duplicated coordinates can
	// produce two identical centroids and therefore an empty cluster. Such a
	// cluster must be reinitialized instead of being returned empty.
	dataset := []Numbers{
		0, 0, 0, 0, 0,
		10,
		20,
	}
	k := 3
	deltaThreshold := 0.01
	iterationThreshold := 100

	expectedClusters := [][]Numbers{
		{0, 0, 0, 0, 0},
		{10},
		{20},
	}

	for seed := int64(0); seed < 100; seed++ {
		rng := rand.New(rand.NewSource(seed))
		clusters, err := Cluster(dataset, k, deltaThreshold, iterationThreshold, rng)
		if err != nil {
			t.Fatalf("seed %d: unexpected error: %v", seed, err)
		}

		if len(clusters) != len(expectedClusters) {
			t.Fatalf("seed %d: expected %d clusters, got %d", seed, len(expectedClusters), len(clusters))
		}

		matched := make([]bool, len(expectedClusters))
		for _, cluster := range clusters {
			if len(cluster) == 0 {
				t.Errorf("seed %d: cluster is empty: %v", seed, clusters)
				continue
			}
			found := false
			for i, expected := range expectedClusters {
				if !matched[i] && slices.Equal(cluster, expected) {
					matched[i] = true
					found = true
					break
				}
			}
			if !found {
				t.Errorf("seed %d: unexpected cluster: %v", seed, cluster)
			}
		}

		for i, m := range matched {
			if !m {
				t.Errorf("seed %d: expected cluster %v not found in %v", seed, expectedClusters[i], clusters)
			}
		}
	}
}
