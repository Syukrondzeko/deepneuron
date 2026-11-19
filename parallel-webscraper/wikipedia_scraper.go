// Command wikipedia_scraper downloads the paragraph text of one English Wikipedia
// article and writes it to output/<article title>.txt.
//
// Usage:
//
//	wikipedia_scraper <URL>
package main

import (
	"fmt"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/gocolly/colly/v2"
)

var invalidFilenameChars = regexp.MustCompile(`[<>:"/\\|?*]`)

// sanitizeTitle replaces characters that are not allowed in filenames.
func sanitizeTitle(title string) string {
	return invalidFilenameChars.ReplaceAllString(strings.TrimSpace(title), "_")
}

// titleFromURL derives a fallback title from the last path segment of the URL.
func titleFromURL(raw string) string {
	u, err := url.Parse(raw)
	if err != nil {
		return ""
	}
	name, err := url.PathUnescape(path.Base(u.Path))
	if err != nil || name == "." || name == "/" {
		return ""
	}
	return strings.ReplaceAll(name, "_", " ")
}

func fail(format string, args ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
	os.Exit(1)
}

func main() {
	if len(os.Args) < 2 {
		fail("Usage: %s <URL>", filepath.Base(os.Args[0]))
	}
	scrapeURL := os.Args[1]

	c := colly.NewCollector(
		colly.AllowedDomains("en.wikipedia.org"),
		colly.UserAgent("deepneuron-wikipedia-scraper/1.0 (educational HPC exercise)"),
	)

	var title string
	var paragraphs []string
	var scrapeErr error

	c.OnHTML("span.mw-page-title-main", func(h *colly.HTMLElement) {
		title = h.Text
	})

	c.OnHTML("p", func(h *colly.HTMLElement) {
		if text := strings.TrimSpace(h.Text); text != "" {
			paragraphs = append(paragraphs, text)
		}
	})

	c.OnRequest(func(r *colly.Request) {
		r.Headers.Set("Accept-Language", "en-US; q=0.9")
		fmt.Printf("Visiting %s\n", r.URL)
	})

	c.OnError(func(r *colly.Response, err error) {
		scrapeErr = fmt.Errorf("HTTP %d: %w", r.StatusCode, err)
	})

	if err := c.Visit(scrapeURL); err != nil {
		fail("Failed to visit %s: %v", scrapeURL, err)
	}
	if scrapeErr != nil {
		fail("Error while scraping %s: %v", scrapeURL, scrapeErr)
	}
	if len(paragraphs) == 0 {
		fail("No paragraph text found at %s", scrapeURL)
	}

	if title == "" {
		title = titleFromURL(scrapeURL)
	}
	name := sanitizeTitle(title)
	if name == "" {
		name = "untitled"
	}

	if err := os.MkdirAll("output", 0o755); err != nil {
		fail("Error creating output directory: %v", err)
	}
	fileName := filepath.Join("output", name+".txt")
	if err := os.WriteFile(fileName, []byte(strings.Join(paragraphs, "\n\n")+"\n"), 0o644); err != nil {
		fail("Error writing %s: %v", fileName, err)
	}

	fmt.Printf("Data successfully written to %s\n", fileName)
}
