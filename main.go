package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"regexp"
	"strings"

	"github.com/google/uuid"
	"github.com/labstack/echo/v4"
	"github.com/labstack/echo/v4/middleware"
	"github.com/qdrant/go-client/qdrant"
	openai "github.com/sashabaranov/go-openai"
	"google.golang.org/grpc"
)

// --- Quadrant/OpenAI Config ---
const (
	embeddingModel     = "text-embedding-ada-002"
	quadrantCollection = "loremipsum" // TODO: set your collection ID
)

var (
	openaiClient *openai.Client
)

// --- Qdrant Client Globals ---
var qdrantClient qdrant.PointsClient

// --- Data Structures ---

// Vector represents the numerical embedding of a text chunk.
type Vector []float64

// Chunk represents a piece of the original document text and its embedding.
type Chunk struct {
	ID     int     `json:"id"`
	Text   string  `json:"text"`
	Vector Vector  `json:"-"`               // Exclude vector from JSON response for brevity
	Score  float64 `json:"score,omitempty"` // Similarity score for query results
}

// --- API Request/Response Structs ---

// ProcessRequest defines the structure for the /process endpoint request.
type ProcessRequest struct {
	Text          string `json:"text"`
	ChunkStrategy string `json:"chunkStrategy"` // "words", "sentences", "paragraphs"
	ChunkSize     int    `json:"chunkSize"`
	Overlap       int    `json:"overlap"`
}

// QueryRequest defines the structure for the /query endpoint request.
type QueryRequest struct {
	Query string `json:"query"`
	TopK  int    `json:"topK"`
}

// --- Main Function ---

func main() {
	openaiKey := os.Getenv("OPENAI_API_KEY")
	if openaiKey == "" {
		panic("Please set OPENAI_API_KEY environment variable")
	}
	openaiClient = openai.NewClient(openaiKey)

	e := echo.New()

	// Middleware
	e.Use(middleware.Logger())
	e.Use(middleware.Recover())
	e.Use(middleware.CORSWithConfig(middleware.CORSConfig{
		AllowOrigins: []string{"*"},
		AllowMethods: []string{http.MethodGet, http.MethodPost},
	}))

	// API Routes
	e.POST("/process", processDocumentHandler)
	e.POST("/query", queryHandler)

	// Qdrant gRPC connection
	conn, err := grpc.Dial("localhost:6334", grpc.WithInsecure()) // adjust host/port as needed
	if err != nil {
		panic("Failed to connect to Qdrant: " + err.Error())
	}
	qdrantClient = qdrant.NewPointsClient(conn)
	collectionsClient := qdrant.NewCollectionsClient(conn)
	err = createCollectionIfNotExists(collectionsClient, quadrantCollection)
	if err != nil {
		log.Fatalf("Collection creation failed: %v", err)
	}

	// Start server
	log.Println("Starting server on :1323")
	e.Logger.Fatal(e.Start(":1323"))
}

func createCollectionIfNotExists(client qdrant.CollectionsClient, name string) error {
	// Check if the collection already exists
	existsResp, err := client.Get(context.Background(), &qdrant.GetCollectionInfoRequest{
		CollectionName: name,
	})
	if err == nil && existsResp != nil {
		log.Printf("Collection '%s' already exists", name)
		return nil
	}

	// Create collection
	_, err = client.Create(context.Background(), &qdrant.CreateCollection{
		CollectionName: name,
		VectorsConfig: &qdrant.VectorsConfig{
			Config: &qdrant.VectorsConfig_Params{
				Params: &qdrant.VectorParams{
					Size:     1536, // Size of the OpenAI Ada embedding
					Distance: qdrant.Distance_Cosine,
				},
			},
		},
	})
	if err != nil {
		return err
	}

	log.Printf("Collection '%s' created", name)
	return nil
}

// --- API Handlers ---

// processDocumentHandler handles chunking the document, generating embeddings, and uploading them to Quadrant.
func processDocumentHandler(c echo.Context) error {
	var req ProcessRequest
	if err := c.Bind(&req); err != nil {
		return c.JSON(http.StatusBadRequest, map[string]string{"error": "Invalid request body"})
	}

	if req.Text == "" {
		return c.JSON(http.StatusBadRequest, map[string]string{"error": "Text content cannot be empty"})
	}

	// 1. Chunk the text based on the selected strategy.
	var chunks []string
	switch req.ChunkStrategy {
	case "words":
		chunks = chunkByWords(req.Text, req.ChunkSize, req.Overlap)
	case "sentences":
		chunks = chunkBySentences(req.Text, req.ChunkSize, req.Overlap)
	case "paragraphs":
		chunks = chunkByParagraphs(req.Text, req.ChunkSize, req.Overlap)
	default:
		return c.JSON(http.StatusBadRequest, map[string]string{"error": "Invalid chunking strategy"})
	}

	uploaded := 0
	for i, chunk := range chunks {
		embedding, err := getEmbedding(chunk)
		if err != nil {
			return c.JSON(http.StatusInternalServerError, map[string]string{"error": fmt.Sprintf("embedding error at chunk %d: %v", i, err)})
		}
		err = uploadToQuadrant(fmt.Sprintf("chunk-%d", i), chunk, embedding)
		if err != nil {
			fmt.Println("Error uploading chunk:", err.Error())
			return c.JSON(http.StatusInternalServerError, map[string]string{"error": fmt.Sprintf("upload error at chunk %d: %v", i, err)})
		}
		uploaded++
	}

	return c.JSON(http.StatusOK, map[string]interface{}{
		"message":         "Document processed and uploaded successfully.",
		"chunks_uploaded": uploaded,
	})
}

// queryHandler handles embedding the query and searching Quadrant for matches.
func queryHandler(c echo.Context) error {
	var req QueryRequest
	if err := c.Bind(&req); err != nil {
		return c.JSON(http.StatusBadRequest, map[string]string{"error": "Invalid request body"})
	}

	if req.Query == "" {
		return c.JSON(http.StatusBadRequest, map[string]string{"error": "Query cannot be empty"})
	}

	embedding, err := getEmbedding(req.Query)
	if err != nil {
		return c.JSON(http.StatusInternalServerError, map[string]string{"error": "Embedding generation failed: " + err.Error()})
	}

	matches, err := searchInQuadrant(embedding, req.TopK)
	if err != nil {
		return c.JSON(http.StatusInternalServerError, map[string]string{"error": "Quadrant search failed: " + err.Error()})
	}

	return c.JSON(http.StatusOK, matches)
}

// --- OpenAI Embedding ---

func getEmbedding(input string) ([]float32, error) {
	resp, err := openaiClient.CreateEmbeddings(
		context.Background(),
		openai.EmbeddingRequest{
			Input: []string{input},
			Model: openai.AdaEmbeddingV2,
		},
	)
	if err != nil {
		println("Error generating embedding:", err.Error())
		return nil, err
	}
	return resp.Data[0].Embedding, nil
}

// --- Qdrant Upload/Search ---

func uploadToQuadrant(tid, text string, vector []float32) error {
	id := uuid.New().String()
	points := []*qdrant.PointStruct{
		{
			Id:      &qdrant.PointId{PointIdOptions: &qdrant.PointId_Uuid{Uuid: id}},
			Vectors: &qdrant.Vectors{VectorsOptions: &qdrant.Vectors_Vector{Vector: &qdrant.Vector{Data: vector}}},
			Payload: map[string]*qdrant.Value{
				"text": {Kind: &qdrant.Value_StringValue{StringValue: text}},
			},
		},
	}
	_, err := qdrantClient.Upsert(context.Background(), &qdrant.UpsertPoints{
		CollectionName: quadrantCollection,
		Points:         points,
	})
	return err
}

func searchInQuadrant(queryEmbedding []float32, topK int) ([]map[string]interface{}, error) {
	if topK <= 0 {
		topK = 5
	}
	res, err := qdrantClient.Search(context.Background(), &qdrant.SearchPoints{
		CollectionName: quadrantCollection,
		Vector:         queryEmbedding,
		Limit:          uint64(topK),
		WithPayload:    &qdrant.WithPayloadSelector{SelectorOptions: &qdrant.WithPayloadSelector_Enable{Enable: true}},
	})
	if err != nil {
		return nil, err
	}
	var matches []map[string]interface{}
	for _, point := range res.Result {
		text := ""
		if v, ok := point.Payload["text"]; ok {
			text = v.GetStringValue()
		}
		matches = append(matches, map[string]interface{}{
			"id":    point.GetId().GetUuid(),
			"score": point.Score,
			"text":  text,
		})
	}
	return matches, nil
}

// --- Text Chunking Logic ---

// chunkByWords splits text by word count with overlap.
func chunkByWords(text string, chunkSize, overlap int) []string {
	words := strings.Fields(text)
	var chunks []string
	if len(words) <= chunkSize {
		return []string{text}
	}

	step := chunkSize - overlap
	if step <= 0 {
		step = 1 // Prevent infinite loops
	}

	for i := 0; i < len(words); i += step {
		end := i + chunkSize
		if end > len(words) {
			end = len(words)
		}
		chunks = append(chunks, strings.Join(words[i:end], " "))
		if end == len(words) {
			break
		}
	}
	return chunks
}

// chunkBySentences splits text by sentence count with overlap.
func chunkBySentences(text string, chunkSize, overlap int) []string {
	// A simple regex for splitting sentences. More robust libraries exist (e.g., NLP libs).
	re := regexp.MustCompile(`[.!?]`)
	sentences := re.Split(text, -1)
	var filteredSentences []string
	for _, s := range sentences {
		if strings.TrimSpace(s) != "" {
			filteredSentences = append(filteredSentences, strings.TrimSpace(s))
		}
	}

	var chunks []string
	if len(filteredSentences) <= chunkSize {
		return []string{text}
	}

	step := chunkSize - overlap
	if step <= 0 {
		step = 1
	}

	for i := 0; i < len(filteredSentences); i += step {
		end := i + chunkSize
		if end > len(filteredSentences) {
			end = len(filteredSentences)
		}
		chunks = append(chunks, strings.Join(filteredSentences[i:end], ". ")+".")
		if end == len(filteredSentences) {
			break
		}
	}
	return chunks
}

// chunkByParagraphs splits text by paragraph count with overlap.
func chunkByParagraphs(text string, chunkSize, overlap int) []string {
	paragraphs := strings.Split(text, "\n\n")
	var filteredParagraphs []string
	for _, p := range paragraphs {
		if strings.TrimSpace(p) != "" {
			filteredParagraphs = append(filteredParagraphs, strings.TrimSpace(p))
		}
	}

	var chunks []string
	if len(filteredParagraphs) <= chunkSize {
		return []string{text}
	}

	step := chunkSize - overlap
	if step <= 0 {
		step = 1
	}

	for i := 0; i < len(filteredParagraphs); i += step {
		end := i + chunkSize
		if end > len(filteredParagraphs) {
			end = len(filteredParagraphs)
		}
		chunks = append(chunks, strings.Join(filteredParagraphs[i:end], "\n\n"))
		if end == len(filteredParagraphs) {
			break
		}
	}
	return chunks
}
