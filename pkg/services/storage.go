package services

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"strings"

	"github.com/jackc/pgx/v5"
	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
)

var pgConn *pgx.Conn
var neo4jDriver neo4j.DriverWithContext

func init() {
	var err error
	pgPassword := os.Getenv("PG_PASSWORD")
	if pgPassword == "" {
		panic("PG_PASSWORD environment variable not set")
	}
	connStr := fmt.Sprintf("postgres://postgres:%s@localhost:5438/postgres?sslmode=disable", pgPassword)
	pgConn, err = pgx.Connect(context.Background(), connStr)
	if err != nil {
		panic(fmt.Sprintf("failed to connect to PostgreSQL: %v", err))
	}

	_, err = pgConn.Exec(context.Background(), "CREATE EXTENSION IF NOT EXISTS vector")
	if err != nil {
		panic(fmt.Sprintf("failed to create pgvector extension: %v", err))
	}

	_, err = pgConn.Exec(context.Background(), `
		CREATE TABLE IF NOT EXISTS documents (
			id SERIAL PRIMARY KEY,
			content TEXT NOT NULL,
			embedding VECTOR(768)
		)
	`)
	if err != nil {
		panic(fmt.Sprintf("failed to create documents table: %v", err))
	}

	_, err = pgConn.Exec(context.Background(), `
		CREATE INDEX IF NOT EXISTS documents_embedding_idx ON documents USING hnsw (embedding vector_l2_ops)
	`)
	if err != nil {
		panic(fmt.Sprintf("failed to create index: %v", err))
	}

	neo4jURI := os.Getenv("NEO4J_URI")
	if neo4jURI != "" {
		neo4jUser := os.Getenv("NEO4J_USER")
		neo4jPassword := os.Getenv("NEO4J_PASSWORD")
		neo4jDriver, err = neo4j.NewDriverWithContext(neo4jURI, neo4j.BasicAuth(neo4jUser, neo4jPassword, ""))
		if err != nil {
			panic(fmt.Sprintf("Failed to connect to Neo4j: %v", err))
		}

		ctx := context.Background()
		session := neo4jDriver.NewSession(ctx, neo4j.SessionConfig{})
		defer session.Close(ctx)
		_, err = session.Run(ctx, `
			CREATE VECTOR INDEX vector_index_token IF NOT EXISTS
			FOR (n:Token) ON (n.embedding)
			OPTIONS {indexConfig: { "vector.dimensions": 768, "vector.similarity_function": "cosine" }}
		`, nil)
		if err != nil {
			panic(fmt.Sprintf("Failed to create Neo4j index: %v", err))
		}
	}
}

func AddDocument(content string, embedding []float32) error {
	var docID int
	err := pgConn.QueryRow(context.Background(), `
		INSERT INTO documents (content, embedding)
		VALUES ($1, $2)
		RETURNING id
	`, content, embedding).Scan(&docID)
	if err != nil {
		return err
	}

	if neo4jDriver != nil {
		return AddTokenAndTriplets(content, embedding, docID)
	}
	return nil
}

type Triplet struct {
	Subject   string
	Predicate string
	Object    string
}

func AddTokenAndTriplets(content string, embedding []float32, docID int) error {
	tokens, triplets := extractTokensAndTriplets(content)

	ctx := context.Background()
	session := neo4jDriver.NewSession(ctx, neo4j.SessionConfig{})
	defer session.Close(ctx)

	_, err := session.Run(ctx, `
		MERGE (d:Document {id: $docID, content: $content})
	`, map[string]interface{}{
		"docID":   docID,
		"content": content,
	})
	if err != nil {
		return err
	}

	for _, token := range tokens {
		tokenEmbedding, err := GetEmbedding(token)
		if err != nil {
			return err
		}
		_, err = session.Run(ctx, `
			MERGE (t:Token {name: $name})
			ON CREATE SET t.embedding = $embedding
			WITH t
			MATCH (d:Document {id: $docID})
			MERGE (t)-[:CONTAINS]->(d)
		`, map[string]interface{}{
			"name":      token,
			"embedding": tokenEmbedding,
			"docID":     docID,
		})
		if err != nil {
			return err
		}
	}

	for _, triplet := range triplets {
		subjectEmb, _ := GetEmbedding(triplet.Subject)
		predicateEmb, _ := GetEmbedding(triplet.Predicate)
		objectEmb, _ := GetEmbedding(triplet.Object)
		_, err := session.Run(ctx, `
			MERGE (s:Token {name: $subject})
			ON CREATE SET s.embedding = $subjectEmb
			MERGE (o:Token {name: $object})
			ON CREATE SET o.embedding = $objectEmb
			MERGE (s)-[r:PREDICATE {name: $predicate}]->(o)
			ON CREATE SET r.embedding = $predicateEmb
		`, map[string]interface{}{
			"subject":      triplet.Subject,
			"subjectEmb":   subjectEmb,
			"predicate":    triplet.Predicate,
			"predicateEmb": predicateEmb,
			"object":       triplet.Object,
			"objectEmb":    objectEmb,
		})
		if err != nil {
			return err
		}
	}

	return nil
}

func extractTokensAndTriplets(content string) ([]string, []Triplet) {
	cmd := exec.Command("python3", "extract_triplets.py")
	cmd.Stdin = strings.NewReader(content)
	var out bytes.Buffer
	cmd.Stdout = &out
	err := cmd.Run()
	if err != nil {
		return strings.Split(content, " "), nil
	}

	var result struct {
		Tokens   []string  `json:"tokens"`
		Triplets []Triplet `json:"triplets"`
	}
	if err := json.Unmarshal(out.Bytes(), &result); err != nil {
		return strings.Split(content, " "), nil
	}
	return result.Tokens, result.Triplets
}

func SearchDocuments(queryEmbedding []float32, useGraph bool) (string, error) {
	var pgContent string
	err := pgConn.QueryRow(context.Background(), `
		SELECT content
		FROM documents
		ORDER BY embedding <-> $1
		LIMIT 1
	`, queryEmbedding).Scan(&pgContent)
	if err != nil {
		return "", fmt.Errorf("failed to search documents in PostgreSQL: %v", err)
	}

	if !useGraph || neo4jDriver == nil {
		return pgContent, nil
	}

	ctx := context.Background()
	session := neo4jDriver.NewSession(ctx, neo4j.SessionConfig{})
	defer session.Close(ctx)

	result, err := session.Run(ctx, `
		CALL db.index.vector.queryNodes('vector_index_token', 10, $queryEmbedding)
		YIELD node AS token, score
		MATCH (token)-[:CONTAINS]->(doc:Document)
		RETURN doc.content AS content
		LIMIT 1
	`, map[string]interface{}{
		"queryEmbedding": queryEmbedding,
	})
	if err != nil {
		return pgContent, fmt.Errorf("failed to search documents in Neo4j: %v", err)
	}

	var neo4jContent string
	if result.Next(ctx) {
		neo4jContent = result.Record().Values[0].(string)
	}

	context := pgContent
	if neo4jContent != "" {
		context += "\nGraph context: " + neo4jContent
	}
	return context, nil
}

func CloseConnection() {
	if pgConn != nil {
		pgConn.Close(context.Background())
	}
	if neo4jDriver != nil {
		neo4jDriver.Close(context.Background())
	}
}
