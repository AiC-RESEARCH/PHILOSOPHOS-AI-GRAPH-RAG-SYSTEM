package services

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
)

const (
	embeddingEndpoint = "https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent?key="
	generateEndpoint  = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key="
)

func GetEmbedding(text string) ([]float32, error) {
	geminiAPIKey := os.Getenv("GEMINI_API_KEY")
	if geminiAPIKey == "" {
		return nil, fmt.Errorf("GEMINI_API_KEY environment variable not set")
	}

	payload := map[string]interface{}{
		"model": "models/embedding-001",
		"content": map[string]interface{}{
			"parts": []map[string]string{
				{"text": text},
			},
		},
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal payload: %v", err)
	}

	resp, err := http.Post(embeddingEndpoint+geminiAPIKey, "application/json", bytes.NewBuffer(body))
	if err != nil {
		return nil, fmt.Errorf("failed to send request to Gemini API: %v", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %v", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("Gemini API embedding request failed with status %s: %s", resp.Status, string(respBody))
	}

	var result struct {
		Embedding struct {
			Values []float32 `json:"values"`
		} `json:"embedding"`
	}
	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %v", err)
	}

	if len(result.Embedding.Values) == 0 {
		return nil, fmt.Errorf("empty embedding returned from Gemini API")
	}

	return result.Embedding.Values, nil
}

func GenerateResponse(query, context string) (string, error) {
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		return "", fmt.Errorf("GEMINI_API_KEY environment variable not set")
	}

	prompt := fmt.Sprintf("Контекст: %s\nВопрос: %s\nОтветь на вопрос, используя контекст.", context, query)
	payload := map[string]interface{}{
		"contents": []map[string]interface{}{
			{
				"parts": []map[string]string{
					{"text": prompt},
				},
			},
		},
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return "", fmt.Errorf("failed to marshal payload: %v", err)
	}

	resp, err := http.Post(generateEndpoint+apiKey, "application/json", bytes.NewBuffer(body))
	if err != nil {
		return "", fmt.Errorf("failed to send request to Gemini API: %v", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read response body: %v", err)
	}

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("Gemini API generate request failed with status %s: %s", resp.Status, string(respBody))
	}

	var result struct {
		Candidates []struct {
			Content struct {
				Parts []struct {
					Text string `json:"text"`
				} `json:"parts"`
			} `json:"content"`
		} `json:"candidates"`
	}
	if err := json.Unmarshal(respBody, &result); err != nil {
		return "", fmt.Errorf("failed to unmarshal response: %v", err)
	}

	if len(result.Candidates) == 0 || len(result.Candidates[0].Content.Parts) == 0 {
		return "", fmt.Errorf("no valid response from Gemini API")
	}

	return result.Candidates[0].Content.Parts[0].Text, nil
}
