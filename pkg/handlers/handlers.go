package handlers

import (
	"io"
	"net/http"
	"strings"
	"sync"

	"github.com/Mukam21/RAG_server-Golang/pkg/services"
	"github.com/gin-gonic/gin"
)

type AddRequest struct {
	Documents []struct {
		Text string `json:"text" binding:"required"`
	} `json:"documents" binding:"required,dive"`
}

func AddDocuments(c *gin.Context) {
	var req AddRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request: " + err.Error()})
		return
	}

	if len(req.Documents) == 0 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Documents list cannot be empty"})
		return
	}

	var errors []string
	var mu sync.Mutex
	var wg sync.WaitGroup

	for _, doc := range req.Documents {
		trimmedText := strings.TrimSpace(doc.Text)
		if len(trimmedText) < 5 {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Document text must be at least 5 characters long"})
			return
		}

		wg.Add(1)
		go func(text string) {
			defer wg.Done()
			embedding, err := services.GetEmbedding(text)
			if err != nil {
				mu.Lock()
				errors = append(errors, err.Error())
				mu.Unlock()
				return
			}
			if err := services.AddDocument(text, embedding); err != nil {
				mu.Lock()
				errors = append(errors, err.Error())
				mu.Unlock()
			}
		}(trimmedText)
	}

	wg.Wait()

	if len(errors) > 0 {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to process some documents", "details": errors})
		return
	}

	c.JSON(http.StatusOK, gin.H{"message": "Documents added successfully"})
}

type QueryRequest struct {
	Query string `json:"query" binding:"required"`
}

func Query(c *gin.Context) {
	var req QueryRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request: " + err.Error()})
		return
	}

	trimmedQuery := strings.TrimSpace(req.Query)
	if len(trimmedQuery) < 3 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Query must be at least 3 characters long"})
		return
	}

	embedding, err := services.GetEmbedding(trimmedQuery)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to get query embedding: " + err.Error()})
		return
	}

	context, err := services.SearchDocuments(embedding)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to search documents: " + err.Error()})
		return
	}

	response, err := services.GenerateResponse(trimmedQuery, context)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to generate response: " + err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"response": response})
}

func UploadDocumentGin(c *gin.Context) {
	file, _, err := c.Request.FormFile("document")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Failed to read file"})
		return
	}
	defer file.Close()

	content, err := io.ReadAll(file)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to read content"})
		return
	}

	embedding, err := services.GetEmbedding(string(content))
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to generate embedding: " + err.Error()})
		return
	}

	if err := services.AddDocument(string(content), embedding); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to save document: " + err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"message": "Document uploaded successfully"})
}
