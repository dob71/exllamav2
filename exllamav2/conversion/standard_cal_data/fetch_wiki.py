import requests
import os

def fetch_random_wikipedia_articles(num_articles, output_file):
    """
    Fetch random Wikipedia articles and append them to the specified file in wiki.utf8 format.
    
    Args:
        num_articles (int): Number of articles to fetch.
        output_file (str): Path to the wiki.utf8 file to append to.
    """
    base_url = "https://en.wikipedia.org/w/api.php"
    articles_fetched = 0
    batch_size = 10  # API allows up to 10 articles per request (rnlimit)

    while articles_fetched < num_articles:
        # Calculate how many articles to fetch in this batch
        remaining = min(batch_size, num_articles - articles_fetched)
        
        # API parameters for random articles
        params = {
            "action": "query",
            "format": "json",
            "list": "random",
            "rnnamespace": 0,  # Main article namespace
            "rnlimit": remaining,
            "rnfilterredir": "nonredirects"  # Exclude redirects
        }
        
        # Fetch random article metadata
        response = requests.get(base_url, params=params)
        if response.status_code != 200:
            print(f"Error fetching articles: HTTP {response.status_code}")
            continue
        
        data = response.json()
        random_pages = data.get("query", {}).get("random", [])
        if not random_pages:
            print("No articles returned in batch")
            continue
        
        # For each random page, fetch its plain text content
        for page in random_pages:
            page_id = page["id"]
            title = page["title"]
            
            # Fetch article content
            content_params = {
                "action": "query",
                "format": "json",
                "pageids": page_id,
                "prop": "extracts",
                "explaintext": True,  # Plain text, no HTML
                "exlimit": 1
            }
            
            content_response = requests.get(base_url, params=content_params)
            if content_response.status_code != 200:
                print(f"Error fetching content for page ID {page_id}")
                continue
            
            content_data = content_response.json()
            page_data = content_data.get("query", {}).get("pages", {}).get(str(page_id), {})
            content = page_data.get("extract", "").strip()
            
            if not content:
                print(f"No content for page ID {page_id} ({title})")
                continue
            
            # Format article in wiki.utf8 style
            article_text = (
                f'<doc id="{page_id}" url="https://en.wikipedia.org/wiki?curid={page_id}" title="{title}">\n'
                f"{title}\n\n"
                f"{content}\n"
                f"</doc>\n"
            )
            
            # Append to file
            try:
                with open(output_file, "a", encoding="utf8") as f:
                    f.write(article_text)
                articles_fetched += 1
                print(f"Appended article {articles_fetched}/{num_articles}: {title} (ID: {page_id})")
            except Exception as e:
                print(f"Error writing article {title}: {e}")
                continue

if __name__ == "__main__":
    # Path to your wiki.utf8 file
    wiki_file = "add_wiki.utf8"  # Update with correct path
    num_articles_to_fetch = 100  # Adjust based on needs
    
    # Verify file exists
    if not os.path.exists(wiki_file):
        print(f"File {wiki_file} does not exist. Creating new file.")
        open(wiki_file, "a", encoding="utf8").close()
    
    # Fetch and append articles
    fetch_random_wikipedia_articles(num_articles_to_fetch, wiki_file)
