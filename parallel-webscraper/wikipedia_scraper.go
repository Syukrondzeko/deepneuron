package main

import (
	"fmt"
	"os"
	"regexp"
	"strings"

	"github.com/gocolly/colly"
)

// SanitizeTitle removes invalid characters from the title for use in filenames
func sanitizeTitle(title string) string {
	re := regexp.MustCompile(`[<>:"/\\|?*]`)
	return re.ReplaceAllString(title, "_")
}

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: go run scraper.go <URL>")
		return
	}

	scrapeUrl := os.Args[1]
	c := colly.NewCollector(colly.AllowedDomains("en.wikipedia.org"))

	var title, extractedText string

	// Extract title from <span class="mw-page-title-main">
	c.OnHTML("span.mw-page-title-main", func(h *colly.HTMLElement) {
		title = h.Text
	})

	// Extract text from <p> tags
	c.OnHTML("p", func(h *colly.HTMLElement) {
		// Extract text from the paragraph
		text := h.Text

		// Clean the text to remove unnecessary whitespaces and newlines
		cleanedText := strings.TrimSpace(text)

		// Append the cleaned text to the extractedText variable
		if extractedText == "" {
			extractedText = cleanedText
		} else {
			extractedText += "\n\n" + cleanedText
		}
	})

	c.OnRequest(func(r *colly.Request) {
		r.Headers.Set("Accept-Language", "en-US; q=0.9")
		fmt.Printf("Visiting %s\n", r.URL)
	})

	c.OnError(func(r *colly.Response, e error) {
		fmt.Printf("Error while scraping %s\n", e.Error())
	})

	err := c.Visit(scrapeUrl)
	if err != nil {
		fmt.Println("Failed to visit the URL:", err)
		return
	}

	// Sanitize the title for use in a filename
	sanitizedTitle := sanitizeTitle(title)
	fileName := "output/" + sanitizedTitle + ".txt"

	// Create output directory if it doesn't exist
	err = os.MkdirAll("output", os.ModePerm)
	if err != nil {
		fmt.Println("Error creating output directory:", err)
		return
	}

	// Open or create the file in the output directory
	file, err := os.Create(fileName)
	if err != nil {
		fmt.Println("Error creating file:", err)
		return
	}
	defer file.Close()

	// Write the extracted text to the file
	_, err = file.WriteString(extractedText)
	if err != nil {
		fmt.Println("Error writing to file:", err)
		return
	}

	fmt.Printf("Data successfully written to %s\n", fileName)
}
